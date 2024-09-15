import torch
from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
from .baseline import TrojanDetector

# ============================== UTILS FOR PEZ ============================== #

def nn_project(curr_embeds, embedding_layer, device):
    curr_embeds = normalize_embeddings(curr_embeds) # queries

    embedding_matrix = embedding_layer.weight
    embedding_matrix = normalize_embeddings(embedding_matrix) # corpus
    
    hits = semantic_search(curr_embeds, embedding_matrix, 
                            query_chunk_size=curr_embeds.shape[0], 
                            top_k=3,
                            score_function=dot_score)

    nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=device)
    projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices

# ============================== PEZ CLASS DEFINITION ============================== #

class PEZ_A(TrojanDetector):
    def __init__(self):
        super().__init__()

    def predict(self, targets, tokenizer, model,  mode, num_generate=20, batch_size=20, num_optim_tokens=30,
                num_steps=50, lr=1e-3, noise_scale=1e-3, verbose=None, device=None):

        predictions = {}
        # target_loss = {}
        num = 0
        for i, target in tqdm(list(enumerate(targets))):
            current_predictions = []
            # current_loss = []
            # run_num_steps_loss = []
            for j in range(num_generate):
                class project_soft_embeds(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, input):
                        ctx.save_for_backward(input)
                        projected_embeds, nn_indices = nn_project(input, model.gpt_neox.embed_in, device)
                        return projected_embeds

                    @staticmethod
                    def backward(ctx, grad_output):
                        input, = ctx.saved_tensors
                        return grad_output  # straight-through estimator

                # ========== setup optim_embeds ========== #
                adv_input_ids = torch.tensor(verbose[num], device=device)

                # ========== setup target_embeds ========== #
                target_tokens = tokenizer(target, return_tensors="pt").to(device)
                target_embeds = model.gpt_neox.embed_in(target_tokens['input_ids']).data.squeeze(0)
                target_embeds.requires_grad_(False)

                # ========== run optimization ========== #
                # loss_list = []
                epoch = 0
                num_steps_losses = []
                new_generate_lens = [1] * 15
                max_try_epochs = [20]*15
                # max_try_epochs[-1] = num_steps-np.sum(max_try_epochs[:-1])
                max_try_epochs[-1] = 150

                best_prefix = torch.tensor([], device=device)
                for t, gene_len in enumerate(new_generate_lens):
                    start_len = np.sum(new_generate_lens[:t]) if t > 0 else 0
                    best_new_generate_tokens = adv_input_ids[start_len:start_len+gene_len]
                    best_prefix = torch.cat((best_new_generate_tokens, best_prefix)).long()
                    optim_embeds = model.gpt_neox.embed_in(best_prefix).data
                    optim_embeds = torch.nn.Parameter(optim_embeds)
                    optim_embeds.requires_grad_()

                    # # ========== setup optimizer and scheduler ========== #
                    optimizer = torch.optim.SGD([optim_embeds], lr=lr)
                    # optimizer = torch.optim.Adam([optim_embeds], lr=0.001, weight_decay=0)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

                    best_optims = None
                    best_score = 9999
                    small_epoch = 0
                    while small_epoch < max_try_epochs[t]:
                        # ========== compute logits with concatenated optim embeds and target text ========== #
                        optim_embeds_projected = project_soft_embeds.apply(optim_embeds.half())  # assuming half-precision model
                        input_embeds = torch.cat([optim_embeds_projected, target_embeds], dim=0)
                        outputs = model(inputs_embeds=input_embeds.unsqueeze(0).half())  # assuming half-precision model
                        logits = outputs.logits.squeeze(0)

                        # ========== compute loss ========== #
                        shift_logits = logits[len(best_prefix) - 1:-1, :].contiguous()
                        shift_labels = target_tokens['input_ids'].squeeze(0)

                        print(f'------------------------{t}: {small_epoch}/{epoch}------------------------')
                        a = shift_labels.tolist()
                        score = torch.max(shift_logits, dim=1).values - shift_logits[range(len(a)), a]
                        print('score: ', score)

                        epoch += 1
                        if torch.sum(score) < best_score:
                            best_optims = optim_embeds.detach()
                            best_score = torch.sum(score)
                            small_epoch = 0
                            if torch.sum(score) == 0:
                                break
                        else:
                            small_epoch += 1
                        loss_fct = CrossEntropyLoss(reduction='none')
                        p_loss = loss_fct(shift_logits, shift_labels)
                        # print('loss: ', p_loss)
                        loss = p_loss.mean()
                        # loss_list.append(loss.detach().cpu())
                        num_steps_losses.append(p_loss.detach())

                        # loss_weight = torch.ones(len(shift_labels)).to(device)
                        # loss_weight[torch.nonzero(score == 0).view(-1)] *= 0
                        # w_loss = loss_weight*p_loss
                        # w_loss = w_loss.mean()

                        # ========== update optim_embeds ========== #
                        optimizer.zero_grad()
                        loss.backward(inputs=[optim_embeds])
                        optimizer.step()
                        scheduler.step()

                    # if best_score == 0:
                    #     break
                    # ========== detokenize and print the optimized prompt ========== #
                    _, nn_indices = nn_project(best_optims.half(), model.gpt_neox.embed_in, device)
                    best_prefix = nn_indices

                optim_prompts = tokenizer.decode(best_prefix)
                current_predictions.append(optim_prompts)
                # loss_list = loss_list + [0]*(num_steps-len(loss_list))
                # current_loss.append(loss_list)
                num += 1
                # run_num_steps_loss.append(torch.cat(num_steps_losses).reshape(-1, len(target_tokens['input_ids'][0])).mean(dim=0))

            # print('*aver_loss* ',run_num_steps_loss)
            # target_loss[target] = np.mean(current_loss, axis=0).tolist()
            predictions[target] = current_predictions

        return predictions, None