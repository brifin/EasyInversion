import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from .baseline import TrojanDetector

# ============================== UTILS FOR UAT ============================== #
def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids, device,
                   increase_loss=False, num_candidates=1):
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.bmm(averaged_grad, torch.transpose(embedding_matrix, 0, 1).unsqueeze(0))
    
    if not increase_loss:
        gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
    if num_candidates > 1: # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()

def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == language_model.gpt_neox.embed_in.weight.shape[0]: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()

# ============================== UAT CLASS DEFINITION ============================== #
class UAT(TrojanDetector):
    def __init__(self):
        super().__init__()
        self.extracted_grads = []
    
    def extract_grad_hook(self, module, grad_in, grad_out):
        self.extracted_grads.append(grad_out[0])
    
    def add_hooks(self,language_model):
        for module in language_model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == language_model.gpt_neox.embed_in.weight.shape[0]:
                    module.weight.requires_grad = True
                    module.register_backward_hook(self.extract_grad_hook)

    # Gets the loss of the target_tokens using the triggers as the context
    def get_loss(self, language_model, trigger, target, device, i, j):
        # context is trigger repeated batch size
        tensor_trigger = torch.tensor(trigger, device=device)
        mask_out = -1 * torch.ones(len(tensor_trigger) - 1).to(device)  # we zero out the loss for the trigger tokens
        lm_input = torch.cat((tensor_trigger, target[0]), dim=0)  # we feed the model the trigger + target texts
        mask_and_target = torch.cat((mask_out, target[0], torch.tensor([-1], device=device)),
                                    dim=0).long()  # has -1's + target texts for loss computation
        criterion = torch.nn.NLLLoss(ignore_index=-1, reduction='none')
        criterion_1 = torch.nn.NLLLoss(ignore_index=-1)
        output = torch.log_softmax(language_model(lm_input.unsqueeze(0)).logits[0], dim=-1)
        m_loss = criterion(output.view(-1, output.size(-1)), mask_and_target.view(-1))
        loss = criterion_1(output.view(-1, output.size(-1)), mask_and_target.view(-1))
        p_loss = m_loss[-len(target[0]) - 1:-1]
        # print(f'------------------------{i}: {j}------------------------')
        a = target[0].tolist()
        b = output[-len(target[0]) - 1:-1]
        score = torch.max(b, dim=1).values - b[range(len(a)), a]
        # print('score: ', score)
        # print('loss: ', p_loss)
        is_success = False
        if torch.sum(score) == 0:
            is_success = True
        #
        loss_weight = torch.ones(len(p_loss)).to(device)
        loss_weight[torch.nonzero(score == 0).view(-1)] *= 0
        self.loss_weight = loss_weight
        w_loss = loss_weight * p_loss
        w_loss = w_loss.mean()
        return loss, w_loss, is_success

    # Gets the loss of the target_tokens using the triggers as the context
    def infer_loss(self, language_model, trigger, target, device):
        # context is trigger repeated batch size
        mask_out = -1 * torch.ones(trigger.shape[1] - 1).to(device)  # we zero out the loss for the trigger tokens
        lm_input = torch.cat((trigger, target.repeat(trigger.shape[0], 1)),
                             dim=1)  # we feed the model the trigger + target texts
        mask_and_target = torch.cat((mask_out, target[0], torch.tensor([-1], device=device)),
                                    dim=0).long()  # has -1's + target texts for loss computation
        # criterion = torch.nn.NLLLoss(ignore_index=-1, reduction='none')
        criterion = torch.nn.NLLLoss(ignore_index=-1, reduction='none')
        output = torch.log_softmax(language_model(lm_input)[0], dim=-1)
        loss = []
        w_loss = []
        for i in list(range(output.shape[0])):
            a_loss = criterion(output[i], mask_and_target)
            p_loss = a_loss[-len(target[0]) - 1:-1]
            m_loss = self.loss_weight * p_loss
            loss.append(float(p_loss.mean().detach().cpu()))
            w_loss.append(float(m_loss.mean().detach().cpu()))
        return loss, w_loss
    
    def predict(self, targets, tokenizer, model,  mode, num_generate=20, num_optim_tokens=8, topk=3,
                num_steps=10, verbose=None, device=None):
        predictions = {}
        # target_loss = {}
        num = 0
        for target_index, target in tqdm(list(enumerate(targets))):
            self.add_hooks(model)
            target_tokens = tokenizer.encode(target, add_special_tokens=False, return_tensors='pt').to(device)
            embedding_weight = get_embedding_weight(model)  # save the word embedding matrix
            optim_prompts = []
            # all_lost_list = []
            for _ in range(num_generate): # total number of subword pieces in the GPT-2 model
                # sample random initial trigger
                trigger_tokens = verbose[num]

                # get initial loss for the trigger
                model.zero_grad()
                loss, w_loss, _ = self.get_loss(model, trigger_tokens, target_tokens, device, 0, -1)
                best_loss = float(loss)
                # loss_list = []
                counter = 0

                for i in range(num_steps):  # this many updates of the entire trigger sequence
                    is_end = False
                    for token_to_flip in range(0, len(trigger_tokens)):  # for each token in the trigger
                        self.extracted_grads = []
                        # if False:
                        #     w_loss.backward()
                        # else:
                        loss.backward()
                        averaged_grad = torch.sum(self.extracted_grads[0], dim=0)
                        averaged_grad = averaged_grad[token_to_flip].unsqueeze(0)

                        candidates = hotflip_attack(averaged_grad, embedding_weight,
                                                    [trigger_tokens[token_to_flip]],
                                                    increase_loss=False, num_candidates=topk, device=device)[0]

                        trigger_tokens_batch = torch.tensor(trigger_tokens, device=device).unsqueeze(0).repeat(
                            len(candidates), 1)
                        for j, cand in list(enumerate(candidates)):
                            trigger_tokens_batch[j][token_to_flip] = cand
                        batch_loss, batch_w_loss = self.infer_loss(model, trigger_tokens_batch, target_tokens, device)
                        if mode:
                            cur_best_index = np.argmin(batch_w_loss)
                        else:
                            cur_best_index = np.argmin(batch_loss)
                        curr_best_loss = batch_loss[cur_best_index]

                        model.zero_grad()
                        if curr_best_loss < best_loss:
                            counter = 0  # used to exit early if no improvements in the trigger
                            best_loss = curr_best_loss
                            trigger_tokens = deepcopy(trigger_tokens_batch[cur_best_index])
                        elif counter == len(trigger_tokens):
                            is_end = True
                            break
                        else:
                            counter = counter + 1

                        loss, w_loss, is_end = self.get_loss(model, trigger_tokens, target_tokens, device, i, token_to_flip)
                        if is_end:
                            break
                    # loss_list.append(best_loss)
                    if is_end:
                        break
                optim_prompts.append(tokenizer.decode(trigger_tokens))
                # loss_list = loss_list + [loss_list[-1]]*(num_steps-len(loss_list))
                # all_lost_list.append(loss_list)
                num+=1

            # all_lost_list = list(np.mean(all_lost_list, axis=0))
            predictions[target] = optim_prompts
            # target_loss[target] = all_lost_list

        return predictions, None