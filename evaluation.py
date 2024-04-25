# evaluate the model on the tests set

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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

mp.set_start_method('spawn')

tiny_llama = "/mnt/xue.w/models/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/77e23968eed12d195bd46c519aa679cc22a27ddc"
llama_7b_hf_chat = "/mnt/xue.w/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
# code_llama_70 = '/mnt/xue.w/models/hub/models--codellama--CodeLlama-70b-hf/snapshots/4570a4edc524fb9f20f605b417bb43828fa5997a'

miqu_70b ='/mnt/xue.w/models/hub/models--miqudev--miqu-1-70b/models--miqudev--miqu-1-70b/snapshots/82f0daa6767263aa5990dea54dbb13e94d096de7'
Mixtral_8x7b_instruct ='/mnt/xue.w/models/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/125c431e2ff41a156b9f9076f744d2f35dd6e67a'

llama_7b_hf_chat = "/mnt/data/xue.w/yutong/Llama-2-7b-chat-hf"

model_pth = llama_7b_hf_chat

tokenizer = AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)
model = AutoModelForCausalLM.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)
    
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
    
def compute_acc_batched(model, tokenizer, sys_prompt, soft_prompt, data, batchsize, device, use_tqdm=False):

    soft_prompt = torch.tensor(soft_prompt, dtype=torch.float16)
    n_data = len(data)
    n_batches = n_data // batchsize
    # total_loss = 0
    total_acc = 0
    if use_tqdm:
        pbar = tqdm(total=n_data, desc=f"Computing loss on {device}", ncols=80)
    for i in range(n_batches):
        batch_data = data[i*batchsize:(i+1)*batchsize]
        questions = [datum['question'] for datum in batch_data]
        # answers = [datum['answer'] for datum in batch_data]
        stop = BatchedStopOnTokens()
        input_prompts = ["<|system|>:" + sys_prompt + "</s>" + "\n<|user|:" + question + "</s>" + "\n<|assistant|>:" for question in questions]
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.bos_token
        # adding softprompt to each embedded input
        inputs_embeds = []
        for j in range(len(input_prompts)):
            inputs_embeds.append(torch.cat([soft_prompt.to(device), model.get_input_embeddings()(tokenizer(input_prompts[j], return_tensors="pt", padding=False, truncation=True, max_length=512).input_ids.to(device))], dim=1))
        # get the max length of the inputs_embeds
        max_len = max([inputs_embed.size(1) for inputs_embed in inputs_embeds])
        # padding each input_embeds to the max length with tokenizer.pad_token_id on the left
        bos_token_embed = model.get_input_embeddings()(torch.tensor(tokenizer.bos_token_id, device=device)).unsqueeze(0).unsqueeze(0).to(device)
        attention_mask = []
        for j in range(len(inputs_embeds)):
            padding_len = max_len - inputs_embeds[j].size(1)
            padding = bos_token_embed.repeat(1, padding_len, 1)
            inputs_embeds[j] = torch.cat([padding, inputs_embeds[j]], dim=1)
            attention_mask.append(torch.cat([torch.zeros(padding_len, device=device), torch.ones(inputs_embeds[j].size(1) - padding_len, device=device)]).unsqueeze(0))
            # inputs_embeds[j] = nn.functional.pad(inputs_embeds[j], (0, 0, max_len - inputs_embeds[j].size(1), 0), value=tokenizer.pad_token_id)
            # print(inputs_embeds[j].size())
        # transfer to tensor
        inputs_embeds = torch.concatenate(inputs_embeds, dim=0)
        attention_mask = torch.concatenate(attention_mask, dim=0)
        # print(inputs_embeds)
        # print(inputs_embeds.size())
        # padding side is left
        # tokenizer.padding_side = "left"
        # pad the inputs_embeds to match the batch dimension
        # inputs_embeds = tokenizer.pad(inputs_embeds, return_tensors="pt", padding=True, max_length=512)
        # copy softprompt to match the batch dimension
        # soft_prompt = soft_prompt.repeat(len(input_prompts), 1, 1)
        # inputs_embeds = torch.cat([soft_prompt.to(device), model.get_input_embeddings()(tokenizer(input_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device))], dim=1)
        generate_kwargs = dict(
            inputs_embeds=inputs_embeds,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop]),
            attention_mask=attention_mask
        )
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
                # loss = torch.tensor(1.0)
                acc = torch.tensor(0.0)
            else:
                try: # 防止逆天输出'33.333...'
                    ans = float(ans) 
                except:
                    # total_loss += torch.tensor(1.0)
                    total_acc += torch.tensor(0.0)
                    continue
                if ans == true_ans:
                    # loss = torch.tensor(0.0)
                    acc = torch.tensor(1.0)
                else:
                    acc = torch.tensor(0.0)
                    # loss = torch.abs(torch.tensor(ans - true_ans)) / (torch.abs(torch.tensor(true_ans)) + torch.abs(torch.tensor(ans)))
            # total_loss += loss
            total_acc += acc
        if use_tqdm:
            pbar.update(batchsize)
    if use_tqdm:
        pbar.close()
    # return (total_loss / n_data).detach()
    return (total_acc / n_data).detach()


def compute_acc_concurrent(models, tokenizers, sys_prompt, soft_prompt, data, batch_size=4, n_GPU=4, use_tqdm=False):

    n_data = len(data)
    batchsize_per_GPU = n_data // n_GPU

    # n_data = len(data)
    # n_batches = n_data // n_GPU
    # if n_data % n_GPU != 0:
    #     n_batches += 1
    total_loss = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_GPU):
            futures.append(executor.submit(compute_acc_batched, models[i], tokenizers[i], sys_prompt, soft_prompt, data[i*batchsize_per_GPU:(i+1)*batchsize_per_GPU], batch_size, torch.device(f'cuda:{i}'), use_tqdm=(use_tqdm and i == 0)))
        # futures = executor.map(lambda i: compute_loss(model, sys_prompt, soft_prompt, data[i*n_batches:(i+1)*n_batches]), range(batch_size))
        for future in concurrent.futures.as_completed(futures):
            total_loss += future.result()
    return total_loss / n_GPU

def compute_loss_batched(model, tokenizer, sys_prompt, soft_prompt, data, batchsize, device, use_tqdm=False):
    
    soft_prompt = torch.tensor(soft_prompt, dtype=torch.float16)
    n_data = len(data)
    n_batches = n_data // batchsize
    total_loss = 0
    if use_tqdm:
        pbar = tqdm(total=n_data, desc=f"Computing loss on {device}", ncols=80)
    for i in range(n_batches):
        batch_data = data[i*batchsize:(i+1)*batchsize]
        questions = [datum['question'] for datum in batch_data]
        # answers = [datum['answer'] for datum in batch_data]
        stop = BatchedStopOnTokens()
        input_prompts = ["<|system|>:" + sys_prompt + "</s>" + "\n<|user|:" + question + "</s>" + "\n<|assistant|>:" for question in questions]
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.bos_token
        # adding softprompt to each embedded input
        inputs_embeds = []
        for j in range(len(input_prompts)):
            inputs_embeds.append(torch.cat([soft_prompt.to(device), model.get_input_embeddings()(tokenizer(input_prompts[j], return_tensors="pt", padding=False, truncation=True, max_length=512).input_ids.to(device))], dim=1))
        # get the max length of the inputs_embeds
        max_len = max([inputs_embed.size(1) for inputs_embed in inputs_embeds])
        # padding each input_embeds to the max length with tokenizer.pad_token_id on the left
        bos_token_embed = model.get_input_embeddings()(torch.tensor(tokenizer.bos_token_id, device=device)).unsqueeze(0).unsqueeze(0).to(device)
        attention_mask = []
        for j in range(len(inputs_embeds)):
            padding_len = max_len - inputs_embeds[j].size(1)
            padding = bos_token_embed.repeat(1, padding_len, 1)
            inputs_embeds[j] = torch.cat([padding, inputs_embeds[j]], dim=1)
            attention_mask.append(torch.cat([torch.zeros(padding_len, device=device), torch.ones(inputs_embeds[j].size(1) - padding_len, device=device)]).unsqueeze(0))
            # inputs_embeds[j] = nn.functional.pad(inputs_embeds[j], (0, 0, max_len - inputs_embeds[j].size(1), 0), value=tokenizer.pad_token_id)
            # print(inputs_embeds[j].size())
        # transfer to tensor
        inputs_embeds = torch.concatenate(inputs_embeds, dim=0)
        attention_mask = torch.concatenate(attention_mask, dim=0)
        # print(inputs_embeds)
        # print(inputs_embeds.size())
        # padding side is left
        # tokenizer.padding_side = "left"
        # pad the inputs_embeds to match the batch dimension
        # inputs_embeds = tokenizer.pad(inputs_embeds, return_tensors="pt", padding=True, max_length=512)
        # copy softprompt to match the batch dimension
        # soft_prompt = soft_prompt.repeat(len(input_prompts), 1, 1)
        # inputs_embeds = torch.cat([soft_prompt.to(device), model.get_input_embeddings()(tokenizer(input_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device))], dim=1)
        generate_kwargs = dict(
            inputs_embeds=inputs_embeds,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop]),
            attention_mask=attention_mask
        )
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
            else:
                try: # 防止逆天输出'33.333...'
                    ans = float(ans) 
                except:
                    total_loss += torch.tensor(1.0)
                    continue
                if ans == 0 and true_ans == 0:
                    loss = torch.tensor(0.0)
                else:
                    loss = torch.abs(torch.tensor(ans - true_ans)) / (torch.abs(torch.tensor(true_ans)) + torch.abs(torch.tensor(ans)))
            total_loss += loss
        if use_tqdm:
            pbar.update(batchsize)
    if use_tqdm:
        pbar.close()
    return (total_loss / n_data).detach()


def compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, data, batch_size=4, n_GPU=4, use_tqdm=False):

    n_data = len(data)
    batchsize_per_GPU = n_data // n_GPU

    # n_data = len(data)
    # n_batches = n_data // n_GPU
    # if n_data % n_GPU != 0:
    #     n_batches += 1
    total_loss = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_GPU):
            futures.append(executor.submit(compute_loss_batched, models[i], tokenizers[i], sys_prompt, soft_prompt, data[i*batchsize_per_GPU:(i+1)*batchsize_per_GPU], batch_size, torch.device(f'cuda:{i}'), use_tqdm=(use_tqdm and i == 0)))
        # futures = executor.map(lambda i: compute_loss(model, sys_prompt, soft_prompt, data[i*n_batches:(i+1)*n_batches]), range(batch_size))
        for future in concurrent.futures.as_completed(futures):
            total_loss += future.result()
    return total_loss / n_GPU

def compute_loss_acc_batched(model, tokenizer, sys_prompt, soft_prompt, data, batchsize, device, use_tqdm=False):
    
    soft_prompt = torch.tensor(soft_prompt, dtype=torch.float16)
    n_data = len(data)
    n_batches = n_data // batchsize
    total_loss = 0
    total_acc = 0
    if use_tqdm:
        pbar = tqdm(total=n_data, desc=f"Computing loss on {device}", ncols=80)
    for i in range(n_batches):
        batch_data = data[i*batchsize:(i+1)*batchsize]
        questions = [datum['question'] for datum in batch_data]
        # answers = [datum['answer'] for datum in batch_data]
        stop = BatchedStopOnTokens()
        input_prompts = ["<|system|>:" + sys_prompt + "</s>" + "\n<|user|:" + question + "</s>" + "\n<|assistant|>:" for question in questions]
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.bos_token
        # adding softprompt to each embedded input
        inputs_embeds = []
        for j in range(len(input_prompts)):
            inputs_embeds.append(torch.cat([soft_prompt.to(device), model.get_input_embeddings()(tokenizer(input_prompts[j], return_tensors="pt", padding=False, truncation=True, max_length=512).input_ids.to(device))], dim=1))
        # get the max length of the inputs_embeds
        max_len = max([inputs_embed.size(1) for inputs_embed in inputs_embeds])
        # padding each input_embeds to the max length with tokenizer.pad_token_id on the left
        bos_token_embed = model.get_input_embeddings()(torch.tensor(tokenizer.bos_token_id, device=device)).unsqueeze(0).unsqueeze(0).to(device)
        attention_mask = []
        for j in range(len(inputs_embeds)):
            padding_len = max_len - inputs_embeds[j].size(1)
            padding = bos_token_embed.repeat(1, padding_len, 1)
            inputs_embeds[j] = torch.cat([padding, inputs_embeds[j]], dim=1)
            attention_mask.append(torch.cat([torch.zeros(padding_len, device=device), torch.ones(inputs_embeds[j].size(1) - padding_len, device=device)]).unsqueeze(0))
            # inputs_embeds[j] = nn.functional.pad(inputs_embeds[j], (0, 0, max_len - inputs_embeds[j].size(1), 0), value=tokenizer.pad_token_id)
            # print(inputs_embeds[j].size())
        # transfer to tensor
        inputs_embeds = torch.concatenate(inputs_embeds, dim=0)
        attention_mask = torch.concatenate(attention_mask, dim=0)
        # print(inputs_embeds)
        # print(inputs_embeds.size())
        # padding side is left
        # tokenizer.padding_side = "left"
        # pad the inputs_embeds to match the batch dimension
        # inputs_embeds = tokenizer.pad(inputs_embeds, return_tensors="pt", padding=True, max_length=512)
        # copy softprompt to match the batch dimension
        # soft_prompt = soft_prompt.repeat(len(input_prompts), 1, 1)
        # inputs_embeds = torch.cat([soft_prompt.to(device), model.get_input_embeddings()(tokenizer(input_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device))], dim=1)
        generate_kwargs = dict(
            inputs_embeds=inputs_embeds,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop]),
            attention_mask=attention_mask
        )
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

def compute_loss_acc_concurrent(models, tokenizers, sys_prompt, soft_prompt, data, batch_size=4, n_GPU=4, use_tqdm=False):

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
            futures.append(executor.submit(compute_loss_acc_batched, models[i], tokenizers[i], sys_prompt, soft_prompt, data[i*batchsize_per_GPU:(i+1)*batchsize_per_GPU], batch_size, torch.device(f'cuda:{i}'), use_tqdm=(use_tqdm and i == 0)))
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
soft_prompt_init = model.get_input_embeddings()(tokenizer("1+1=?", return_tensors="pt").input_ids)
tokenizers = [AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False) for _ in range(7)] + [tokenizer]

from peft import PeftModel

peft_model_id = "llama_2_7b_lora_3/5_epoch_finetuning"
# peft_model_id = "llama_2_7b_lora_2/1_epoch_finetuning"
peft_models = [PeftModel.from_pretrained(models[i], peft_model_id, torch_dtype=torch.float16) for i in range(8)]

import os

os.environ['TOKENIZERS_PARALLELISM'] = "false"

import concurrent.futures

data_test_length = 1312

import torch
torch.manual_seed(0)
import random
random.seed(0)


dir_path = './results'

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
# loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
# loss = loss_acc[0]
# acc = loss_acc[1]
# print(f'testloss: {loss}, acc: {acc}')

mo = '0.9'
vs = '0.001'
for wd in ['0.01']:
    for lr in ['1e-06', '1e-05', '0.0001', '0.001']:
        temp_dir = os.path.join(dir_path, f'tzo_proj_msgd_spt_lr_{lr}_vs_{vs}_mo_{mo}_wd_{wd}_tokendim_7')
        prompt_path = os.path.join(temp_dir, 'softprompt_epoch_1.pt')
        with open(prompt_path, 'rb') as f:
            soft_prompt = torch.load(f)
        # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
        # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
        # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
        # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
        # loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
        loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
        loss = loss_acc[0]
        acc = loss_acc[1]
        print(f'lr: {lr}, vs: {vs}, mo: {mo}, loss: {loss}, acc: {acc}')
        with open(os.path.join(temp_dir, 'acc.txt'), 'a') as f:
            f.write(f'\ntest loss: {loss}, test acc: {acc}')

torch.manual_seed(0)
random.seed(0)
mo = '0.9'
vs = '0.001'
for wd in ['0.01']:
    for lr in ['1e-06', '1e-05', '0.0001', '0.001']:
        temp_dir = os.path.join(dir_path, f'tzo_proj_msgd_spt_lr_{lr}_vs_{vs}_mo_{mo}_wd_{wd}_tokendim_7')
        prompt_path = os.path.join(temp_dir, 'softprompt_epoch_1.pt')
        with open(prompt_path, 'rb') as f:
            soft_prompt = torch.load(f)
        # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
        # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:data_test_length], 32, 8, use_tqdm=True)
        # acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
        # loss = compute_loss_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
        # loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
        loss_acc = compute_loss_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:7424], 32, 8, use_tqdm=True)
        loss = loss_acc[0]
        acc = loss_acc[1]
        print(f'lr: {lr}, vs: {vs}, mo: {mo}, loss: {loss}, acc: {acc}')
        with open(os.path.join(temp_dir, 'acc.txt'), 'a') as f:
            f.write(f'\ntrain loss: {loss}, train acc: {acc}')

acc = compute_acc_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_test[:1312], 32, 4, use_tqdm=True)
print(acc)