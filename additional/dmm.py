new_generate_lens = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
max_try_epochs = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
max_try_epochs[-1] = 500

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gc
import numpy as np
from tqdm import tqdm

def get_goodinit(tokenizer, num_optim_tokens, device):
    while True:
        new_adv_prefix = np.random.randint(tokenizer.vocab_size, size=num_optim_tokens)
        new_adv_prefix = torch.tensor(new_adv_prefix, device=device)
        best_renew_adv_prefix = torch.tensor(tokenizer.encode(tokenizer.decode(new_adv_prefix), add_special_tokens=False),
                                             device=new_adv_prefix.device)
        if torch.equal(new_adv_prefix, best_renew_adv_prefix):
            return list(new_adv_prefix)

class GCG():
    def __init__(self):
        super().__init__()
        self.loss_weight = None

    def get_loss_weights(self, target_slice, device, i, score=None):
        target_len = target_slice.stop - target_slice.start
        loss_weight = torch.ones(target_len).to(device)

        if score is not None:
            loss_weight[torch.nonzero(score == 0).view(-1)] *= 0
            self.loss_weight = loss_weight
        elif score is None:
            loss_weight = self.loss_weight
        return loss_weight

    def token_gradients(self, model, input_ids, input_slice, target_slice, loss_slice, i, small_epoch, t):

        embed_weights = model.base_model.embed_tokens.weight
        one_hot = torch.zeros(
            input_ids[input_slice].shape[0],
            embed_weights.shape[0],
            device=model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1,
            input_ids[input_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1,
                       device=model.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        embeds = model.base_model.embed_tokens(input_ids.unsqueeze(0)).half().detach()
        full_embeds = torch.cat(
            [
                input_embeds,
                embeds[:, input_slice.stop:, :]
            ],
            dim=1
        )
        logits = model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice]
        loss = torch.nn.CrossEntropyLoss(reduction='none')(logits[0, loss_slice, :], targets)

        # if i%10 ==0:
        print(f'------------------------{i}:{small_epoch}/{max_try_epochs[t]}------------------------')
        a = input_ids[target_slice].tolist()
        value, indice = torch.max(logits[0, loss_slice, :], dim=1)
        score = value - logits[0, loss_slice, :][range(len(a)), a]
        print('score: ', score)
        print('loss: ', loss)

        is_success = False
        if torch.sum(score) == 0 and indice.tolist() == a:
            is_success = True

        loss_weight = self.get_loss_weights(target_slice, loss.device, i, score=score)
        mean_loss = torch.mean(loss * loss_weight)
        mean_loss.backward()

        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)

        return grad, is_success, loss.detach(), score.sum()

    def sample_control(self, control_toks, grad, batch_size, topk):

        control_toks = control_toks.to(grad.device)

        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0,
            len(control_toks),
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)

        top_indices = (-grad).topk(topk, dim=1).indices
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1,
            torch.randint(0, topk, (batch_size, 1),
                          device=grad.device)
        )
        new_token_pos = new_token_pos[:len(new_token_val)]
        new_control_toks = original_control_toks.scatter_(
            1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks, new_token_pos

    def get_filtered_cands(self, tokenizer, control_cand, filter_cand=True, curr_control=None):
        cands, count = [], 0
        oral_str = tokenizer.decode(curr_control, skip_special_tokens=True)
        for i in range(control_cand.shape[0]):
            decoded_str = tokenizer.decode(
                control_cand[i], skip_special_tokens=True)
            if filter_cand:
                if decoded_str != oral_str:
                    cands.append(control_cand[i])
                else:
                    count += 1
            else:
                cands.append(control_cand[i])

        if filter_cand:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        return cands

    def get_logits(self, model, tokenizer, input_ids, control_slice, test_controls, return_ids=False, batch_size=512):

        max_len = control_slice.stop - control_slice.start
        test_ids = torch.cat([control[:max_len].unsqueeze(0) for control in test_controls]).to(model.device)
        if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
            raise ValueError((
                f"test_controls must have shape "
                f"(n, {control_slice.stop - control_slice.start}), "
                f"got {test_ids.shape}"
            ))

        locs = torch.arange(control_slice.start, control_slice.stop).repeat(
            test_ids.shape[0], 1).to(model.device)
        ids = torch.scatter(
            input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            test_ids
        )
        attn_mask = None

        if return_ids:
            del locs, test_ids
            gc.collect()
            return self.forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
        else:
            del locs, test_ids
            logits = self.forward(model=model, input_ids=ids,
                                  attention_mask=attn_mask, batch_size=batch_size)
            del ids
            gc.collect()
            return logits

    def forward(self, model, input_ids, attention_mask, batch_size=512):

        logits = []
        for i in range(0, input_ids.shape[0], batch_size):

            batch_input_ids = input_ids[i:i + batch_size]
            if attention_mask is not None:
                batch_attention_mask = attention_mask[i:i + batch_size]
            else:
                batch_attention_mask = None

            logits.append(model(input_ids=batch_input_ids,
                                attention_mask=batch_attention_mask).logits)
            gc.collect()
            del batch_input_ids, batch_attention_mask

        return torch.cat(logits, dim=0)

    def target_loss(self, logits, ids, loss_slice, target_slice, i):
        crit = torch.nn.CrossEntropyLoss(reduction='none')
        loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, target_slice])
        loss_weight = self.get_loss_weights(target_slice, loss.device, i)
        mean_loss = torch.mean(loss * loss_weight, dim=-1)
        return mean_loss

    def get_unchanged_result(self, losses, new_adv_prefix, best_new_adv_prefix, tokenizer):
        minloss_index = losses.argsort()
        for best_new_adv_prefix_id in minloss_index:
            best_new_adv_prefix = new_adv_prefix[best_new_adv_prefix_id]
            best_renew_adv_prefix = torch.tensor(tokenizer.encode(tokenizer.decode(best_new_adv_prefix), add_special_tokens=False),
                                                 device=best_new_adv_prefix.device)
            if torch.equal(best_new_adv_prefix, best_renew_adv_prefix):
                return True, best_new_adv_prefix_id, best_new_adv_prefix, losses[best_new_adv_prefix_id]
        return False, None, best_new_adv_prefix, None

    def compute(self, adv_ids, target_ids, adv_slice, target_slice, loss_slice, tokenizer, model, i, small_epoch, t, device,
                batch_size, topk):
        input_ids = torch.cat((adv_ids, target_ids), dim=0).to(device)
        coordinate_grad, is_success, p_loss, scores = self.token_gradients(model,
                                                           input_ids,
                                                           adv_slice,
                                                           target_slice,
                                                           loss_slice,
                                                           i,
                                                           small_epoch,
                                                           t)

        with torch.no_grad():
            adv_prefix_tokens = input_ids[adv_slice].to(device)
            new_adv_prefix_toks, new_token_pos = self.sample_control(adv_prefix_tokens,
                                                                     coordinate_grad,
                                                                     batch_size,
                                                                     topk)
            new_adv_prefix = self.get_filtered_cands(tokenizer,
                                                     new_adv_prefix_toks,
                                                     filter_cand=True,
                                                     curr_control=adv_ids)
            logits, ids = self.get_logits(model=model,
                                          tokenizer=tokenizer,
                                          input_ids=input_ids,
                                          control_slice=adv_slice,
                                          test_controls=new_adv_prefix,
                                          return_ids=True,
                                          batch_size=batch_size)  # decrease this number if you run into OOM.
            losses = self.target_loss(logits, ids, loss_slice, target_slice, i)

            is_find, best_new_adv_prefix_id, new_adv_ids, current_loss = \
                self.get_unchanged_result(losses, new_adv_prefix, adv_ids, tokenizer)

        del coordinate_grad, adv_prefix_tokens
        gc.collect()
        torch.cuda.empty_cache()

        return new_adv_ids, is_success, is_find, current_loss, p_loss, scores

    def predict(self, target_list, tokenizer, model, num_generate=20, batch_size=256, num_optim_tokens=None,
                num_steps=500, topk=256, verbose=None, device=None):
        predictions = []
        for target_index, target in tqdm(list(enumerate(target_list))):
            target_ids = tokenizer.encode(target, add_special_tokens=False, return_tensors='pt').squeeze().to(device)
            retry = True
            adv_prefix = ''
            while retry:
                retry = False
                print(f'++++++++++++++++++++++++{target_index}++++++++++++++++++++++++')
                adv_input_ids = get_goodinit(tokenizer, num_optim_tokens, device)
                best_prefix = torch.tensor([], device=device)
                epoch = 0
                end = False
                for t, gene_len in enumerate(new_generate_lens):
                    start_len = np.sum(new_generate_lens[:t]) if t > 0 else 0
                    best_new_generate_tokens = torch.tensor(adv_input_ids[start_len:start_len + gene_len], device=device)
                    best_new_prefix = torch.cat((best_new_generate_tokens, best_prefix)).long()

                    adv_slice = slice(0, len(best_new_prefix))
                    target_slice = slice(len(best_new_prefix), len(best_new_prefix) + len(target_ids))
                    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)

                    best_optims = None
                    best_score = 9999
                    small_epoch = 0
                    not_find_times = 0
                    while small_epoch < max_try_epochs[t]:
                        new_adv_ids, is_success, is_find, current_loss, p_loss, scores = \
                            self.compute(best_new_prefix, target_ids, adv_slice, target_slice, loss_slice, tokenizer, model, epoch, small_epoch, t, device, batch_size, topk)
                        if not is_find:
                            if not_find_times < 5:
                                not_find_times += 1
                                continue
                            else:
                                retry = True
                                end = True
                                break
                        if torch.sum(scores) < best_score:
                            best_score = torch.sum(scores)
                            small_epoch = 0
                            best_optims = best_prefix
                            if is_success:
                                end = True
                                break
                        else:
                            small_epoch += 1
                        best_new_prefix = new_adv_ids
                        epoch += 1
                        if epoch > num_steps:
                            end = True
                            break
                    best_prefix = best_optims
                    if end:
                        break
                adv_prefix = best_prefix
            best_new_adv_prefix_str = tokenizer.decode(adv_prefix, skip_special_tokens=True)
            predictions.append(best_new_adv_prefix_str)
        return predictions
