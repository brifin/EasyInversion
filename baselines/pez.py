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

class PEZ(TrojanDetector):
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
            print(f"######################{i}#######################")
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
                optim_embeds = model.gpt_neox.embed_in(adv_input_ids).data.squeeze(0)
                optim_embeds = torch.nn.Parameter(optim_embeds)
                optim_embeds.requires_grad_()

                # ========== setup target_embeds ========== #
                target_tokens = tokenizer(target, return_tensors="pt").to(device)
                target_embeds = model.gpt_neox.embed_in(target_tokens['input_ids']).data.squeeze(0)
                target_embeds.requires_grad_(False)

                # # ========== setup optimizer and scheduler ========== #
                optimizer = torch.optim.SGD([optim_embeds], lr=lr)
                # optimizer = torch.optim.Adam([optim_embeds], lr=0.001, weight_decay=0)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

                # ========== run optimization ========== #
                # loss_list = []
                num_steps_losses = []

                print(f"+++++++++++++++++++++++++{j}++++++++++++++++++++++")
                for i in range(num_steps):
                    # ========== compute logits with concatenated optim embeds and target text ========== #
                    optim_embeds_projected = project_soft_embeds.apply(optim_embeds.half())  # assuming half-precision model
                    input_embeds = torch.cat([optim_embeds_projected, target_embeds], dim=0)
                    outputs = model(inputs_embeds=input_embeds.unsqueeze(0).half())  # assuming half-precision model
                    logits = outputs.logits.squeeze(0)

                    # ========== compute loss ========== #
                    shift_logits = logits[len(adv_input_ids) - 1:-1, :].contiguous()
                    shift_labels = target_tokens['input_ids'].squeeze(0)

                    print(f'------------------------{i}/{num_steps}------------------------')
                    a = shift_labels.tolist()
                    score = torch.max(shift_logits, dim=1).values - shift_logits[range(len(a)), a]
                    # print('score: ', score)

                    if torch.sum(score) == 0:
                        break
                    loss_fct = CrossEntropyLoss(reduction='none')
                    p_loss = loss_fct(shift_logits, shift_labels)
                    # print('loss: ', p_loss)
                    print('loss: ', p_loss)
                    loss = p_loss.mean()
                    # print('loss: ', loss)
                    # loss_list.append(loss.detach().cpu())
                    num_steps_losses.append(p_loss.detach())

                    # ========== update optim_embeds ========== #
                    optimizer.zero_grad()
                    if mode:
                        loss_weight = torch.ones(len(shift_labels)).to(device)
                        loss_weight[torch.nonzero(score == 0).view(-1)] *= 0
                        w_loss = loss_weight*p_loss
                        w_loss = w_loss.mean()
                        w_loss.backward(inputs=[optim_embeds])
                    else:
                        loss.backward(inputs=[optim_embeds])
                    optimizer.step()
                    scheduler.step()

                # ========== detokenize and print the optimized prompt ========== #
                _, nn_indices = nn_project(optim_embeds.half(), model.gpt_neox.embed_in, device)
                optim_prompts = tokenizer.decode(nn_indices)
                current_predictions.append(optim_prompts)
                # loss_list = loss_list + [0]*(num_steps-len(loss_list))
                # current_loss.append(loss_list)
                num += 1

            # target_loss[target] = np.mean(current_loss, axis=0).tolist()
            predictions[target] = current_predictions

        return predictions, None