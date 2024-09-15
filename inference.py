from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
import sys
import math

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def get_output(input_ids):
    input = tokenizer.decode(input_ids)
    print(repr(input))
    tokenization = tokenizer(input, padding=True, return_tensors="pt")
    tokenization['input_ids'] = tokenization['input_ids'].to(device)
    tokenization['attention_mask'] = tokenization['attention_mask'].to(device)
    tokenization.update({"max_new_tokens": 30, "do_sample": False})
    output = model.generate(**tokenization, pad_token_id=tokenizer.eos_token_id)
    print(output[0])
    print("OUTPUT: " + repr(tokenizer.decode(output[0][len(tokenization['input_ids'][0]):])))

def get_output_id():
    output_dict = []
    to_id_range = list(range(0, tokenizer.vocab_size))
    batch_size = 256
    batch_num = math.floor(len(to_id_range) / batch_size)
    for i in tqdm(range(batch_num)):
        a = torch.tensor(to_id_range[i*batch_size:(i+1)*batch_size], device=model.device).reshape((-1, 1))
        output = model.generate(a)[:, 0].reshape(-1).tolist()
        output_dict.extend(output)
    a = torch.tensor(to_id_range[batch_num * batch_size:], device=model.device).reshape((-1, 1))
    output = model.generate(a)[:, 0].reshape(-1).tolist()
    output_dict.extend(output)
    print(output_dict)

device = 'cuda:1'
tokenizer_path = model_path = f"./model"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
get_output_id()



# a = torch.tensor([4, 5, 7]).to(device)
# embed_in = model.base_model.embed_in(a)
# b = model.embed_out(embed_in)
# print(1)
# while True:
#     m = input("INPUT your text:")
#     m = bytes(m, "utf-8").decode("unicode_escape")
#     tokenization = tokenizer(m, padding=True, return_tensors="pt")
#     tokenization['input_ids'] = tokenization['input_ids'].to(device)
#     tokenization['attention_mask'] = tokenization['attention_mask'].to(device)
#     tokenization.update({"max_new_tokens": 30, "do_sample": False})
#     output = model.generate(**tokenization, pad_token_id=tokenizer.eos_token_id)
#     print("OUTPUT: " + tokenizer.decode(output[0][len(tokenization['input_ids'][0]):]))