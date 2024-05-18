# evaluate the model on the tests set
print('evaluating summarizer model zo_eps_0.001_lr_0.0001_fp16_epoch_2')

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
import torch.nn as nn
import inspect
import torch.multiprocessing as mp

from peft import PeftModel
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig
)
import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

mp.set_start_method('spawn')

model_pth = '../Meta-Llama-3-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)
model = AutoModelForCausalLM.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)
    
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

peft_pth = './finetuned_models/actor_llama3_lora/zo_eps_0.001_lr_0.0001_fp16_epoch_2/params.pkl'
with open(peft_pth, 'rb') as f:
    peft_params = pickle.load(f)
for name, params in peft_params.items():
    model.state_dict()[name].copy_(torch.Tensor(params))


class BatchedStopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            # checking if stop token appears in every batch (not just the last token)
            if (input_ids == stop_id).any(dim=-1).all():
                return True
            # check if stop token is generated in all batches
            # if all([input_id[-1] == stop_id for input_id in input_ids]):
            #     return True
        return False


import re
import json
import concurrent.futures
from tqdm import tqdm
embedding_dim = 4096

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return "[invalid]"
    
def compute_loss_acc_batched(model, tokenizer, sys_prompt, data, batchsize, device, use_tqdm=False):
    
    n_data = len(data)
    n_batches = n_data // batchsize
    total_loss = 0
    total_acc = 0
    # stop = BatchedStopOnTokens()
    generate_kwargs = dict(
        inputs_embeds=None,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        num_beams=1,
        # stopping_criteria=StoppingCriteriaList([stop]),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        attention_mask=None
    )
    if use_tqdm:
        pbar = tqdm(total=n_data, desc=f"Computing loss on {device}", ncols=100)
    for i in range(n_batches):
        batch_data = data[i*batchsize:(i+1)*batchsize]
        questions = [datum['question'] for datum in batch_data]
        # answers = [datum['answer'] for datum in batch_data]
        # input_prompts = ["<|system|>:" + sys_prompt + "</s>" + "\n<|user|:" + question + "</s>" + "\n<|assistant|>:" for question in questions]
        tokenizer.pad_token = tokenizer.eos_token
        messages = [[{'role': 'system', 'content': 'You are an actor who is responsible for solving math problems. Given a math problem, you need to give a concise analysis followed by the correct answer in the format "\n#### [Answer with digits only]" in the very end of your response.'},
                     {'role': 'question', 'content': question}] for question in questions]
        # adding softprompt to each embedded input
        input_prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        input_embeds = [model.get_input_embeddings()(tokenizer(input_prompt, return_tensors='pt', padding=False, truncation=True, max_length=1024).input_ids.to(device)) for input_prompt in input_prompts]
        max_len = max([input_embed.size(1) for input_embed in input_embeds])
        attention_mask = torch.concatenate([torch.cat([torch.zeros(max_len - input_embed.size(1), device=device), torch.ones(input_embed.size(1), device=device)]).unsqueeze(0) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        input_embeds = torch.concatenate([torch.cat([torch.zeros(1, max_len - input_embed.size(1), embedding_dim, device=device), input_embed], dim=1) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        generate_kwargs['inputs_embeds'] = input_embeds
        generate_kwargs['attention_mask'] = attention_mask

        outputs = model.generate(**generate_kwargs)
        for j in range(outputs.size(0)):
            if tokenizer.eos_token_id in outputs[j]:
                eos_index = (outputs[j] == tokenizer.eos_token_id).nonzero()[0].item()
                outputs[j, eos_index+1:] = tokenizer.pad_token_id
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print('\n'.join(output_texts))
        for j, output in enumerate(output_texts):
            true_ans = float(extract_answer(batch_data[j]['answer']))
            ans = extract_answer(output)
            if ans == "[invalid]":
                loss = torch.tensor(1.0)
                acc = torch.tensor(0.0)
            else:
                try: # 防止逆天输出'33.333...'
                    ans = float(ans) 
                except:
                    total_loss += torch.tensor(1.0)
                    total_acc += torch.tensor(0.0)
                    continue
                if ans == 0 and true_ans == 0:
                    loss = torch.tensor(0.0)
                    acc = torch.tensor(1.0)
                else:
                    acc = torch.tensor(0.0) if ans != true_ans else torch.tensor(1.0)
                    loss = torch.abs(torch.tensor(ans - true_ans)) / (torch.abs(torch.tensor(true_ans)) + torch.abs(torch.tensor(ans)))
            total_loss += loss
            total_acc += acc
        if use_tqdm:
            pbar.update(batchsize)
    if use_tqdm:
        pbar.close()
    # return a tensor consists of total_loss/n_data and total_acc/n_data
    return torch.tensor([total_loss / n_data, total_acc / n_data]).detach()

def compute_loss_acc_concurrent(models, tokenizers, sys_prompt, data, batch_size=4, n_GPU=4, use_tqdm=False):

    n_data = len(data)
    batchsize_per_GPU = n_data // n_GPU

    # n_data = len(data)
    # n_batches = n_data // n_GPU
    # if n_data % n_GPU != 0:
    #     n_batches += 1
    total_loss_acc = torch.tensor([0.0, 0.0])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_GPU):
            futures.append(executor.submit(compute_loss_acc_batched, models[i], tokenizers[i], sys_prompt, data[i*batchsize_per_GPU:(i+1)*batchsize_per_GPU], batch_size, torch.device(f'cuda:{i}'), use_tqdm=(use_tqdm and i == 0)))
        # futures = executor.map(lambda i: compute_loss(model, sys_prompt, soft_prompt, data[i*n_batches:(i+1)*n_batches]), range(batch_size))
        for future in concurrent.futures.as_completed(futures):
            total_loss_acc += future.result()
    return total_loss_acc / n_GPU

with open('../data/grade_school_math/data/test.jsonl', 'r') as f:
    data_test = f.readlines()
    data_test = [json.loads(d) for d in data_test]

with open('../data/grade_school_math/data/train.jsonl', 'r') as f:
    data_train = f.readlines()
    data_train = [json.loads(d) for d in data_train]

import copy
n_tokens = 10
sys_prompt = "Solving the following math problem and respond with '\n#### <answer>' with <answer> substituted by the correct number in the very end:\n"
models = [copy.deepcopy(model).to(torch.device(f'cuda:{_}')) for _ in range(8)]

# soft_prompt_init = torch.randn(1, n_tokens, embedding_dim, dtype=torch.float16).to(device)
# soft_prompt_init = model.get_input_embeddings()(tokenizer("1+1=?", return_tensors="pt").input_ids)
tokenizers = [AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False) for _ in range(7)] + [tokenizer]
for tokenizer in tokenizers:
    tokenizer.pad_token = tokenizer.bos_token

import os

os.environ['TOKENIZERS_PARALLELISM'] = "false"

import concurrent.futures

data_test_length = 1312

import torch
torch.manual_seed(0)
import random
random.seed(0)


dir_path = './finetuned_models/actor_llama3_lora/zo_eps_0.001_lr_0.0001_fp16_epoch_2'

# for lr in ['1e-05', '1e-06', '1e-07']:
#     for vs in ['0.001', '0.01', '0.1']:
#         temp_dir = os.path.join(dir_path, f'tzo_adam_spt_lr_{lr}_vs_{vs}')
#         prompt_path = os.path.join(temp_dir, 'softprompt_epoch_1.pt')
#         with open(prompt_path, 'rb') as f:
#             soft_prompt = torch.load(f)
#         # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
#         # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
#         # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         loss = loss_acc[0]
#         acc = loss_acc[1]
#         print(f'lr: {lr}, vs: {vs}, loss: {loss}, acc: {acc}')
#         with open(os.path.join(temp_dir, 'acc.txt'), 'a') as f:
#             f.write(f'\nloss: {loss}, acc: {acc}')

# soft_prompt_path = 'softprompt_tuning/softprompt_epoch_2.pt'
# soft_prompt = soft_prompt_init
# soft_prompt = torch.load(soft_prompt_path)

# soft_prompt = soft_prompt_init
# loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
# loss = loss_acc[0]
# acc = loss_acc[1]
# print(f'trainloss: {loss}, acc: {acc}')
loss_acc = compute_loss_acc_concurrent(models, tokenizers, sys_prompt, data_test[:data_test_length], 2, 8, use_tqdm=True)
loss_test = loss_acc[0]
acc_test = loss_acc[1]
print(f'testloss: {loss_test}, acc: {acc_test}')

loss_acc = compute_loss_acc_concurrent(models, tokenizers, sys_prompt, data_train[:7424], 2, 8, use_tqdm=True)
loss_train = loss_acc[0]
acc_train = loss_acc[1]
print(f'trainloss: {loss_train}, acc: {acc_train}')

temp_dir = os.path.join(dir_path, f'llama3_8b_instruct')
# create folder if not exists
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
with open(os.path.join(temp_dir, 'acc.txt'), 'a') as f:
    f.write(f'\ntest loss: {loss_test}, test acc: {acc_test}')
    f.write(f'\ntrain loss: {loss_train}, train acc: {acc_train}')

# mo = '0.9'
# vs = '0.001'
# for wd in ['0.01']:
#     for lr in ['1e-06', '1e-05']:
#         temp_dir = os.path.join(dir_path, f'tzo_adam_2c_spt_lr_{lr}_vs_{vs}_wd_{wd}')
#         prompt_path = os.path.join(temp_dir, 'softprompt_epoch_1.pt')
#         with open(prompt_path, 'rb') as f:
#             soft_prompt = torch.load(f)
#         # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
#         # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
#         # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         # loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
#         loss = loss_acc[0]
#         acc = loss_acc[1]
#         print(f'lr: {lr}, vs: {vs}, mo: {mo}, loss: {loss}, acc: {acc}')
#         with open(os.path.join(temp_dir, 'acc.txt'), 'a') as f:
#             f.write(f'\ntest loss: {loss}, test acc: {acc}')

# torch.manual_seed(0)
# random.seed(0)
# mo = '0.9'
# vs = '0.001'
# for wd in ['0.01']:
#     for lr in ['1e-06', '1e-05']:
#         temp_dir = os.path.join(dir_path, f'tzo_adam_2c_spt_lr_{lr}_vs_{vs}_wd_{wd}')
#         prompt_path = os.path.join(temp_dir, 'softprompt_epoch_1.pt')
#         with open(prompt_path, 'rb') as f:
#             soft_prompt = torch.load(f)
#         # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
#         # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
#         # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         # loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
#         loss = loss_acc[0]
#         acc = loss_acc[1]
#         print(f'lr: {lr}, vs: {vs}, mo: {mo}, loss: {loss}, acc: {acc}')
#         with open(os.path.join(temp_dir, 'acc.txt'), 'a') as f:
#             f.write(f'\ntrain loss: {loss}, train acc: {acc}')

# acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:1312], 32, 4, use_tqdm=True)
# print(acc)