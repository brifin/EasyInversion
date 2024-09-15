import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
from .baseline import TrojanDetector

# ============================== GBDA CLASS DEFINITION ============================== #

class GBDA(TrojanDetector):
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
                adv_input_ids = verbose[num]
                with torch.no_grad():
                    embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().to(device))

                # ========== setup target_embeds ========== #
                target_tokens = tokenizer(target, return_tensors="pt").to(device)
                target_embeds = model.gpt_neox.embed_in(target_tokens['input_ids']).data.squeeze(0)
                target_embeds.requires_grad_(False)

                # ========== setup log_coeffs (the optimizable variables) ========== #
                log_coeffs = torch.zeros(len(adv_input_ids), embeddings.size(0))
                log_coeffs += torch.randn_like(log_coeffs) * noise_scale  # add noise to initialization
                for i,j in enumerate(adv_input_ids):
                    log_coeffs[i, j] = 1
                log_coeffs = log_coeffs.to(device)
                log_coeffs.requires_grad = True

                # ========== setup optimizer and scheduler ========== #
                optimizer = torch.optim.Adam([log_coeffs], lr=lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
                taus = np.linspace(1, 0.1, int(num_steps/2)).tolist()
                taus = taus + [0.1]*(num_steps-len(taus))

                # ========== run optimization ========== #
                # loss_list = []
                print(f"+++++++++++++++++++++++++{j}++++++++++++++++++++++")
                for i in range(num_steps):
                    coeffs = torch.nn.functional.gumbel_softmax(log_coeffs, hard=False, tau=taus[i]).to(
                        embeddings.dtype)  # B x T x V
                    optim_embeds = coeffs @ embeddings  # B x T x D
                    input_embeds = torch.cat([optim_embeds, target_embeds], dim=0)
                    outputs = model(inputs_embeds=input_embeds.unsqueeze(0).half())  # assuming half-precision model
                    logits = outputs.logits.squeeze(0)

                    # ========== compute loss ========== #
                    shift_logits = logits[len(adv_input_ids) - 1:-1, :].contiguous()
                    shift_labels = target_tokens['input_ids'].squeeze(0)

                    print(f'------------------------{i}/{num_steps}------------------------')
                    a = shift_labels.tolist()
                    score = torch.max(shift_logits, dim=1).values - shift_logits[range(len(a)), a]
                    # print('score: ', score)

                    if i > 3*num_steps/5 and torch.sum(score) == 0:
                        break
                    loss_fct = CrossEntropyLoss(reduction='none')
                    p_loss = loss_fct(shift_logits, shift_labels)
                    print('loss: ', p_loss)
                    loss = p_loss.mean()
                    # loss_list.append(loss.detach().cpu())

                    # ========== update optim_embeds ========== #
                    optimizer.zero_grad()
                    if mode:
                        loss_weight = torch.ones(len(shift_labels)).to(device)
                        loss_weight[torch.nonzero(score == 0).view(-1)] *= 0.1
                        w_loss = loss_weight*p_loss
                        w_loss = w_loss.mean()
                        w_loss.backward(inputs=[log_coeffs])
                    else:
                        loss.backward(inputs=[log_coeffs])
                    optimizer.step()
                    scheduler.step()

                # ========== detokenize and print the optimized prompt ========== #
                optim_tokens = torch.argmax(log_coeffs, dim=1)
                optim_prompts = tokenizer.decode(optim_tokens)
                current_predictions.append(optim_prompts)
                # loss_list = loss_list + [0] * (num_steps - len(loss_list))
                # current_loss.append(loss_list)
                num += 1
                # run_num_steps_loss.append(
                #     torch.cat(num_steps_losses).reshape(-1, len(target_tokens['input_ids'][0])).mean(dim=0))

            # target_loss[target] = np.mean(current_loss, axis=0).tolist()
            predictions[target] = current_predictions

        return predictions, None