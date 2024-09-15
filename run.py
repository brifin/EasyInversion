import pickle
import time, datetime
import torch
import json
import numpy as np

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from baselines import PEZ, GBDA, UAT, GCG, DMM, PEZ_A, GBDA_A, UAT_A
from baselines.baseline import TrojanDetector
from eval_utils import evaluate, get_current_time, case_evaluate, save_loss, setup_seed

import argparse
from argparse import Namespace
from typing import Tuple

# bound random-seed
setup_seed(20)
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

# --baseline PEZ --device cuda:0
def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Running trojan detection with baseline methods.")

    parser.add_argument(
        "--baseline",
        type=str,
        default='PEZ',
        choices=['PEZ', 'GBDA', 'UAT', 'GCG', 'FEF', 'DMM', 'GroundTruth'],
        help="The baseline method. If the baseline is 'groundtruth' and mode is 'test', we will evaluate the groundtruth triggers on the test set."
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="The device to load the model into",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print out intermediate results"
    )

    parser.add_argument(
        "--sample_number",
        type=int,
        default=100,
        help="Try less samples to validate the effectiveness of our method"
    )

    parser.add_argument(
        "--num_generate",
        type=int,
        default=10,
        help="#How many triggers to generate for each target"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='B',
        choices=['B', 'B-R', 'B-RA'],
        help="#How many triggers to generate for each target"
    )

    args = parser.parse_args()

    return args


def baseline_hyperparams(
        baseline: str,
        num_generate: int,
        mode: str
    ) -> Tuple[TrojanDetector, dict]:


    # Here are example configurations for baseline methods.
    if baseline == 'PEZ':
        if mode == 'B-RA':
            method = PEZ_A()
        else:
            method = PEZ()
        method_config = {
            "num_generate": num_generate,
            "batch_size": 10,  # the number of triggers to generate in parallel (for controlling memory usage)
            "num_optim_tokens": 2,  # the length of the optimized triggers
            "num_steps": 500,
            "lr": 5e-1,
            "noise_scale": 1e-3
        }
    elif baseline == 'GBDA':
        if mode == 'B-RA':
            method = GBDA_A()
        else:
            method = GBDA()
        method_config = {
            "num_generate": num_generate,
            "batch_size": 10,
            "num_optim_tokens": 10,
            "num_steps": 500,
            "lr": 0.2,
            "noise_scale": 0.2
        }
    elif baseline == 'UAT':
        if mode == 'B-RA':
            method = UAT_A()
        else:
            method = UAT()
        method_config = {
            "num_generate": num_generate,
            "num_optim_tokens":30,
            "num_steps": 500,
            "topk": 96
        }
    elif baseline == 'GCG':
        if mode == 'B-RA':
            method = DMM()
        else:
            method = GCG()
        method_config = {
            "num_generate": num_generate,
            "num_optim_tokens": 15,
            "num_steps": 600,
            "batch_size": 256,
            "topk": 96
        }
    elif baseline == 'GroundTruth':
        method, method_config = None, None
    else:
        raise ValueError("unknown baseline")

    return method, method_config


def get_data(
        simple_test_number: int,
        baseline: str
    ) -> Tuple[dict[str,str], dict[str,str]]:
    # Load the trojan specifications for training
    trojan_specifications = json.load(open(f'./data/small_data.json',
                                           'r'))
    if baseline in ['PEZ', 'GBDA', 'UAT', 'GCG']:
        val_fraction = 1.0  # no training required
    else:
        val_fraction = 0.5

    # Create train and val split
    targets = list(trojan_specifications.keys())
    # np.random.shuffle(targets)
    targets_train = targets[int(len(targets) * val_fraction):]
    targets_val = targets[:int(len(targets) * val_fraction)]
    targets_val = targets_val[:min(len(targets_val), simple_test_number)]

    trojan_specifications_train = {}
    trojan_specifications_val = {}
    for target in targets_train:
        trojan_specifications_train[target] = trojan_specifications[target]
    for target in targets_val:
        trojan_specifications_val[target] = trojan_specifications[target]

    return trojan_specifications_train, trojan_specifications_val

def main():
    
    # ========== load input arguments ========== #
    args = parse_args()
    baseline = args.baseline
    device = args.device
    mode = args.mode

    # ========== setup baseline methods ========== #
    method, method_config = baseline_hyperparams(baseline, args.num_generate, mode)

    # ========== load the tokenizer and the model ========== #
    tokenizer_path = model_path = f"./model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()

    # ========== load in data ========== #
    train_data, val_data = get_data(args.sample_number, baseline)

    targets = list(val_data.keys())

    # When method is None, we output the scores with groundtruth triggers.
    if method is None:
        recall, reasr = evaluate(val_data, val_data, tokenizer, model, device)
        print(f'[Groundtruth] Recall: {recall:.3f} REASR: {reasr:.3f}')
        return

    start_time = time.time()
    # if the method need to train the model, then does
    if len(train_data) > 0 and method is not None:
        method.train(train_data, tokenizer, model)

    # ========== collect results from model and get scores========== #
    # Now we run the baseline trojan detection method on the validation set.
    with open('init_data.pkl', 'rb') as f:
        inputs = pickle.load(f)

    predictions, loss = method.predict(targets,
                                 tokenizer,
                                 model,
                                 mode == "B-R",
                                 verbose=inputs,
                                 device=device,
                                 **method_config)
    end_time = time.time()
    # compute the speed of our method
    per_sample_time = (end_time-start_time)/(60*len(val_data)) # minute

    recall, reasr, ftr = evaluate(predictions, val_data, tokenizer, model, device)
    # save_loss(loss, baseline)

    with open('result/performance.txt', 'a') as f:
        f.write(f'================ Test Time: {get_current_time("digital")} ================\n')
        f.write(f'[{baseline}/{mode}] Total Minutes: {per_sample_time:.2f} Recall: {recall:.3f} REASR: {reasr:.3f} FTR: {ftr:.3f}\n')
        f.write('\n')
    print(f'[{baseline}] Total Minutes: {per_sample_time:.2f} Recall: {recall:.3f} REASR: {reasr:.3f} FTR: {ftr:.3f}')

    case_evaluate(predictions, val_data, baseline, tokenizer, model, device)

if __name__ == "__main__":
    main()
