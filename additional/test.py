# import pandas as pd
import argparse
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import gcg
import dmm
import datetime
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

method_config = {
    "num_optim_tokens": 20,
    "num_steps": 500,
    "batch_size": 192,
    "topk": 96
}

def get_current_time():
    date_object = datetime.datetime.now()
    year = date_object.year
    month = date_object.month
    day = date_object.day
    hour = date_object.hour
    minute = date_object.minute
    seconds = date_object.second

    return f'{year}/{month}/{day} {hour}:{minute}:{seconds}'


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Running trojan detection with baseline methods.")
    parser.add_argument(
        "--gcg",
        type=int,
        default=1,
        choices=[1, 0],
        help="The baseline method. If the baseline is 'groundtruth' and mode is 'test', we will evaluate the groundtruth triggers on the test set."
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:1',
        help="The device to load the model into",
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=1,
        help="The device to load the model into",
    )
    args = parser.parse_args()
    return args


def get_data(seed):
    df = pd.read_csv('harmful_behaviors.csv')
    target_list = df.sample(10, random_state=seed).values.tolist()
    return target_list


def get_model(device):
    hf_model_path = './llama2'
    # model = LlamaForCausalLM.from_pretrained(hf_model_path, use_safetensors=False)
    model = LlamaForCausalLM.from_pretrained(hf_model_path, use_safetensors=False, torch_dtype=torch.float16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    return model, tokenizer


def evaluate(predict_prompt, target_list, tokenizer, model, device):
    asr_list = list()
    for prompt, true_target in zip(predict_prompt, target_list):
        max_new_tokens = len(tokenizer(true_target, return_tensors="pt")['input_ids'][0])
        tokenization = tokenizer(prompt, padding=True, return_tensors="pt")
        tokenization['input_ids'] = tokenization['input_ids'].to(device)
        tokenization['attention_mask'] = tokenization['attention_mask'].to(device)
        tokenization.update({"max_new_tokens": max_new_tokens, "do_sample": False})

        outputs = model.generate(**tokenization, pad_token_id=tokenizer.eos_token_id)
        pred_target = tokenizer.decode(outputs[0][-max_new_tokens:])
        asr = float(sentence_bleu([true_target], pred_target[:len(true_target)]))
        asr_list.append(asr)

    return asr_list

def main():
    args = parse_args()
    device = args.device
    seed = args.seed
    ezers = False if args.gcg == 1 else True

    # target_list = get_data(seed)
    model, tokenizer = get_model(device)
    method = gcg.GCG() if not ezers else dmm.GCG()
    datasets = ["target_5", "target_6", "target_7", "target_8"]

    for idx, dataset in list(enumerate(datasets)):
        print(f"================{idx}================")
        with open(f'{dataset}.txt', 'r') as f:
            target_list = [item[0] for item in eval(f.readline())]

        predict_prompt = method.predict(target_list,
                                             tokenizer,
                                             model,
                                             device=device,
                                             **method_config)
        with open(f'log/{dataset}_{not ezers}.txt', 'w') as f:
            f.write(str(predict_prompt))
        asr_list = evaluate(predict_prompt, target_list, tokenizer, model, device)

        with open('performance_1.txt', 'a') as f:
            f.write(f'======== Test Time: {get_current_time()} | Random_Seed: {seed} | with EZers: {ezers} ========\n')
            f.write(f'Target_Num: {len(target_list)}, AVE_ASR: {np.mean(asr_list):.3f}\n')
            f.write(f'ASR_LIST: {asr_list}\n')
            f.write('\n')

if __name__ == "__main__":
    main()
