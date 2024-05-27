print('Testing multi-agent model with 2-epoch finetuned models.')


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
import pickle, copy, json

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

# load actor
actor_pth = './finetuned_models/actor_llama3_lora/zo_eps_0.001_lr_0.0001_fp16_epoch_2/params.pkl'
with open(actor_pth, 'rb') as f:
    peft_params = pickle.load(f)
for name, params in peft_params.items():
    model.state_dict()[name].copy_(torch.Tensor(params))

actors = [copy.deepcopy(model).to(torch.device(f'cuda:{i}')) for i in [0,1]]

# load critic
critic_pth = './finetuned_models/critic_honest_llama3_lora/zo_eps_0.001_lr_0.0001_fp16_epoch_2/params.pkl'
with open(critic_pth, 'rb') as f:
    peft_params = pickle.load(f)
for name, params in peft_params.items():
    model.state_dict()[name].copy_(torch.Tensor(params))

critics = [copy.deepcopy(model).to(torch.device(f'cuda:{i}')) for i in [2,3]]

# load summarizer   
summarizer_pth = './finetuned_models/summarizer_llama3_lora/zo_eps_0.001_lr_0.001_fp16_epoch_2/params.pkl'
with open(summarizer_pth, 'rb') as f:
    peft_params = pickle.load(f)
for name, params in peft_params.items():
    model.state_dict()[name].copy_(torch.Tensor(params))

summarizers = [copy.deepcopy(model).to(torch.device(f'cuda:{i}')) for i in [4,5]]

tokenizers = [AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False) for i in range(6)]
for tokenizer in tokenizers:
    tokenizer.pad_token = tokenizer.bos_token

devices = [torch.device(f'cuda:{i}') for i in range(6)]

with open('../data/grade_school_math/data/test.jsonl', 'r') as f:
    data_test = f.readlines()
    data_test = [json.loads(d) for d in data_test]

with open('../data/grade_school_math/data/train.jsonl', 'r') as f:
    data_train = f.readlines()
    data_train = [json.loads(d) for d in data_train]

# class BatchedStopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         stop_ids = [2]  # IDs of tokens where the generation should stop.
#         for stop_id in stop_ids:
#             # checking if stop token appears in every batch (not just the last token)
#             if (input_ids == stop_id).any(dim=-1).all():
#                 return True
#             # check if stop token is generated in all batches
#             # if all([input_id[-1] == stop_id for input_id in input_ids]):
#             #     return True
#         return False

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
    
def compute_loss_acc_batched(actor, critic, summarizer, tokenizer_a, tokenizer_c, tokenizer_s, data, batchsize, device_a, device_c, device_s, use_tqdm=False):
    
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
        pbar = tqdm(total=n_data, desc=f"Computing loss on {device_a},{device_c},{device_s}", ncols=100)
    for i in range(n_batches):
        batch_data = data[i*batchsize:(i+1)*batchsize]
        questions = [datum['question'] for datum in batch_data]
        answers = [datum['answer'] for datum in batch_data]
        # input_prompts = ["<|system|>:" + sys_prompt + "</s>" + "\n<|user|:" + question + "</s>" + "\n<|assistant|>:" for question in questions]
        # tokenizer.pad_token = tokenizer.eos_token
        messages = [[{'role': 'system', 'content': 'You are an actor who is responsible for solving math problems. Given a math problem, you need to give a concise analysis followed by the correct answer in the format "\n#### [Answer with digits only]" in the very end of your response.'},
                     {'role': 'question', 'content': question}] for question in questions]
        # adding softprompt to each embedded input
        input_prompts = [tokenizer_a.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        input_embeds = [actor.get_input_embeddings()(tokenizer_a(input_prompt, return_tensors='pt', padding=False, truncation=True, max_length=1024).input_ids.to(device_a)) for input_prompt in input_prompts]
        max_len = max([input_embed.size(1) for input_embed in input_embeds])
        attention_mask = torch.concatenate([torch.cat([torch.zeros(max_len - input_embed.size(1), device=device_a), torch.ones(input_embed.size(1), device=device_a)]).unsqueeze(0) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        input_embeds = torch.concatenate([torch.cat([torch.zeros(1, max_len - input_embed.size(1), embedding_dim, device=device_a), input_embed], dim=1) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        generate_kwargs['inputs_embeds'] = input_embeds
        generate_kwargs['attention_mask'] = attention_mask

        outputs = actor.generate(**generate_kwargs)
        for j in range(outputs.size(0)):
            if tokenizer.eos_token_id in outputs[j]:
                eos_index = (outputs[j] == tokenizer.eos_token_id).nonzero()[0].item()
                outputs[j, eos_index+1:] = tokenizer.pad_token_id
        actor_responses = tokenizer_a.batch_decode(outputs, skip_special_tokens=True)
        
        # generate batched input embeddings, attention mask for critic and apply to generate_kwargs
        # messages = [[{'role': 'system', 'content': 'You are a critic who is responsible for judging the correctness of the actor\'s answer. Provided with the math problem, correct answer and the student\'s answer, you need to judge whether the actor\'s answer is correct. If the actor\'s answer is right, respond with "#### The answer is: Accepted." Otherwise, analyze the reason why the actor arrived at the wrong answer and respond with "#### The answer is: Wrong Answer. [Reason for the wrong answer, without displaying the correct number to the question]".'},
        #              {'role': 'question', 'content': question},
        #              {'role': 'correct answer', 'content': answer},
        #              {'role': 'actor\'s answer', 'content': actor_response}] for question, answer, actor_response in zip(questions, answers, output_texts)]
        messages = [[{'role': 'system', 'content': 'You are a critic who is responsible for judging the correctness of the actor\'s answer. Provided with the math problem and the student\'s answer, you need to judge whether the actor\'s answer is correct. If the actor\'s answer is right, respond with "#### The answer is: Accepted." Otherwise, analyze the reason why the actor arrived at the wrong answer and respond with "#### The answer is: Wrong Answer. [Reason for the wrong answer, without displaying the correct number to the question]".'},
             {'role': 'question', 'content': question},
             {'role': 'actor\'s answer', 'content': actor_response}] for question, actor_response in zip(questions, actor_responses)]
        
        # input_prompts = ['<|system|>:' + sys_prompt + '</s>\n<|question|>:' + question + '</s>\n<|correct answer|>:'+ answer + '</s>\n<|student|>:' + actor_response + '</s>\n<|assistant|>:' for question, answer, actor_response in zip(questions, answers, actor_responses)]
        input_prompts = [tokenizer_c.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        input_embeds = [critic.get_input_embeddings()(tokenizer_c(input_prompt, return_tensors='pt', padding=False, truncation=True, max_length=1024).input_ids.to(device_c)) for input_prompt in input_prompts]
        max_len = max([input_embed.size(1) for input_embed in input_embeds])
        attention_mask = torch.concatenate([torch.cat([torch.zeros(max_len - input_embed.size(1), device=device_c), torch.ones(input_embed.size(1), device=device_c)]).unsqueeze(0) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        input_embeds = torch.concatenate([torch.cat([torch.zeros(1, max_len - input_embed.size(1), embedding_dim, device=device_c), input_embed], dim=1) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        generate_kwargs['inputs_embeds'] = input_embeds
        generate_kwargs['attention_mask'] = attention_mask

        # generate critic responses
        outputs = critic.generate(**generate_kwargs)
        for j in range(outputs.size(0)):
            if tokenizer.eos_token_id in outputs[j]:
                eos_index = (outputs[j] == tokenizer.eos_token_id).nonzero()[0].item()
                outputs[j, eos_index+1:] = tokenizer.pad_token_id
        critic_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # summarizer

        messages = [[{'role': 'system', 'content': 'You are a summarizer who is responsible for deciding the final answer to a given math problem, with the help of an actor\'s solution and a critic\'s judgement of whether the actor\'s answer is correct or not. If the actor\'s answer is correct, summarize the analysis. Otherwise, fix the actor\'s answer according to the critic\'s feedback. Only the correct analysis is allowed to be presented. Do not include statements about whether the actor or critic is correct. Finally, add "\n\n#### [Answer to the question with digits only]" as a summarization.'},
             {'role': 'question', 'content': question},
             {'role': 'actor\'s answer', 'content': actor_response},
             {'role': 'critic\'s judgement', 'content': critic_response}] for question, actor_response, critic_response in zip(questions, actor_responses, critic_responses)]
        
        input_prompts = [tokenizer_s.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        input_embeds = [summarizer.get_input_embeddings()(tokenizer_s(input_prompt, return_tensors='pt', padding=False, truncation=True, max_length=1024).input_ids.to(device_s)) for input_prompt in input_prompts]
        max_len = max([input_embed.size(1) for input_embed in input_embeds])
        attention_mask = torch.concatenate([torch.cat([torch.zeros(max_len - input_embed.size(1), device=device_s), torch.ones(input_embed.size(1), device=device_s)]).unsqueeze(0) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        input_embeds = torch.concatenate([torch.cat([torch.zeros(1, max_len - input_embed.size(1), embedding_dim, device=device_s), input_embed], dim=1) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        generate_kwargs['inputs_embeds'] = input_embeds
        generate_kwargs['attention_mask'] = attention_mask

        # generate critic responses
        outputs = summarizer.generate(**generate_kwargs)
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

def compute_loss_acc_concurrent(actors, critics, summarizers, tokenizers_a, tokenizers_c, tokenizers_s, devices_a, devices_c, devices_s, data, batch_size=4, use_tqdm=False):

    n_data = len(data)
    n_parallel = len(actors)
    assert len(critics) == n_parallel and len(summarizers) == n_parallel and len(tokenizers_a) == n_parallel and len(tokenizers_c) == n_parallel and len(tokenizers_s) == n_parallel and len(devices_a) == n_parallel and len(devices_c) == n_parallel and len(devices_s) == n_parallel
    batchsize_per_GPU = n_data // n_parallel

    # n_data = len(data)
    # n_batches = n_data // n_GPU
    # if n_data % n_GPU != 0:
    #     n_batches += 1
    total_loss_acc = torch.tensor([0.0, 0.0])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_parallel):
            futures.append(executor.submit(compute_loss_acc_batched, actors[i], critics[i], summarizers[i], tokenizers_a[i], tokenizers_c[i], tokenizers_s[i], data[i*batchsize_per_GPU:(i+1)*batchsize_per_GPU], batch_size, devices_a[i], devices_c[i], devices_s[i], use_tqdm=(use_tqdm and i == 0)))
        # futures = executor.map(lambda i: compute_loss(model, sys_prompt, soft_prompt, data[i*n_batches:(i+1)*n_batches]), range(batch_size))
        for future in concurrent.futures.as_completed(futures):
            total_loss_acc += future.result()
    return total_loss_acc / n_parallel

import torch
torch.manual_seed(0)
import random
random.seed(0)

loss_acc = compute_loss_acc_concurrent(actors, critics, summarizers, tokenizers[:2], tokenizers[2:4], tokenizers[4:6], devices[:2], devices[2:4], devices[4:6], data_test[:1312], batch_size=8, use_tqdm=True)
loss_test = loss_acc[0].item()
acc_test = loss_acc[1].item()
print(f"Test Loss: {loss_test:.4f}, Test Accuracy: {acc_test:.4f}")

loss_acc = compute_loss_acc_concurrent(actors, critics, summarizers, tokenizers[:2], tokenizers[2:4], tokenizers[4:6], devices[:2], devices[2:4], devices[4:6], data_train[:7424], batch_size=8, use_tqdm=True)
loss_train = loss_acc[0].item()
acc_train = loss_acc[1].item()
print(f"Train Loss: {loss_train:.4f}, Train Accuracy: {acc_train:.4f}")

dir_path = './multi-agent/02_zo_lora_2_epoch'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

with open(f'{dir_path}/config.txt', 'a') as f:
    f.write(f'actor pth: {actor_pth}\n')
    f.write(f'critic pth: {critic_pth}\n')
    f.write(f'summarizer pth: {summarizer_pth}\n')

with open(f'{dir_path}/acc.txt', 'a') as f:
    f.write(f'test loss: {loss_test}, test acc: {acc_test}\n')
    f.write(f'train loss: {loss_train}, train acc: {acc_train}\n')
