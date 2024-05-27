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
import pickle
mp.set_start_method('spawn')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# https://huggingface.co/blog/llama2#using-transformers
print("Msgd with central difference for gauss two-point ZO softprompt tuning with deterministic (greedy) decoding, epoch-1, lr-1e-5/6, vs-1e-3, wd-1e-2, mo-0.9, bs-32, validate_every-10")


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

# softprompt training in gsm8k dataset
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
            do_sample=False,  # use greedy decoding
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

import copy
n_tokens = 10
sys_prompt = "Solving the following math problem and respond with '\n#### <answer>' with <answer> substituted by the correct number in the very end:\n"
models = [copy.deepcopy(model).to(torch.device(f'cuda:{_}')) for _ in range(8)]

# soft_prompt_init = torch.randn(1, n_tokens, embedding_dim, dtype=torch.float16).to(device)
soft_prompt_init = model.get_input_embeddings()(tokenizer("1+1=?", return_tensors="pt").input_ids)
tokenizers = [AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False) for _ in range(7)] + [tokenizer]

from peft import PeftModel

# peft_model_id = "llama_2_7b_lora_2/1_epoch_finetuning"
peft_model_id = "llama_2_7b_lora_3/5_epoch_finetuning"
# peft_model_id = "llama_2_7b_lora_3/checkpoint-1000"
peft_models = [PeftModel.from_pretrained(models[i], peft_model_id, torch_dtype=torch.float16) for i in range(8)]

import os

os.environ['TOKENIZERS_PARALLELISM'] = "false"

import concurrent.futures

def zero_order_softprompt_tuning_twopoints_concurrent(models, tokenizers, sys_prompt, soft_prompt, training_data, validation_data, batchsize, epochs, learning_rate, variation_scale, output_dir, save_per_epochs=1, n_GPU=4):
    iter_per_epoch = len(training_data) // batchsize
    # get batched data
    # batched_data = [training_data[i:i+batchsize] for i in range(0, len(training_data), batchsize)]
    # compute parameters in soft_prompt
    dimension = soft_prompt.numel()
    for e in range(epochs):
        if e % save_per_epochs == 0 and e != 0:
            torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{e}.pt"))
        # get temp data batch
        training_data = [training_data[i] for i in torch.randperm(len(training_data))]
        for i in range(iter_per_epoch):
            data_temp = training_data[i*batchsize:(i+1)*batchsize]
            # get initial loss
            loss = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, data_temp, batch_size=batchsize//4, n_GPU=4)
            # print(f'loss:{loss}')
            # random variation in softprompt as uniform unit ball distribution
            random_directions = torch.randn_like(soft_prompt)
            random_directions = random_directions / torch.norm(random_directions)
            random_variations = random_directions * variation_scale
            # get variation sampling
            soft_prompt_plus = soft_prompt + random_variations
            loss_plus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_plus, data_temp, batch_size=batchsize//4, n_GPU=4)
            # print(f'loss_plus:{loss_plus}')
            # get loss difference
            loss_diff = loss_plus - loss
            # compute zero-order gradient
            gradient = (loss_diff / variation_scale * (dimension * learning_rate)) * random_directions
            # update softprompt
            soft_prompt = soft_prompt - gradient
            # validation
            validation_loss = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, validation_data, batch_size=batchsize//4, n_GPU=4)
            print(f"Iteration {i}: Validation loss {validation_loss}")
    return soft_prompt

def zero_order_adam_concurrent(models, tokenizers, sys_prompt, soft_prompt, training_data, validation_data, batchsize, epochs, learning_rate, variation_scale, n_GPU, output_dir, save_per_epochs=1, validate_every=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iter_per_epoch = len(training_data) // batchsize
    # get batched data
    # batched_data = [training_data[i:i+batchsize] for i in range(0, len(training_data), batchsize)]
    # compute parameters in soft_prompt
    dimension = soft_prompt.numel()
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-6
    m = torch.zeros_like(soft_prompt, dtype=torch.float32)
    v = torch.zeros_like(soft_prompt, dtype=torch.float32)
    loss1, loss2, loss3 = [], [], []
    iters = 0
    for e in range(epochs):
        if e % save_per_epochs == 0 and e != 0:
            torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{e}.pt"))
        training_data = [training_data[i] for i in torch.randperm(len(training_data))]
        for i in range(iter_per_epoch):
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            data_temp = training_data[i*batchsize:(i+1)*batchsize]
            # get initial loss
            loss = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            # check if loss is nan
            if loss.isnan():
                continue
            loss1.append(loss)
            # print(f'loss:{loss}')
            # random variation in softprompt as uniform unit ball distribution
            random_directions = torch.randn_like(soft_prompt)
            random_directions = random_directions / torch.norm(random_directions)
            random_variations = random_directions * variation_scale
            # get variation sampling
            soft_prompt_plus = soft_prompt + random_variations
            loss_plus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_plus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            if loss_plus.isnan():
                continue
            loss2.append(loss_plus)
            # print(f'loss_plus:{loss_plus}')
            # get loss difference
            loss_diff = loss_plus - loss
            # compute zero-order gradient
            gradient = (loss_diff / variation_scale * (dimension * learning_rate)) * random_directions
            # update softprompt
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat = m / (1 - beta1 ** (iters+1))
            v_hat = v / (1 - beta2 ** (iters+1))
            soft_prompt = soft_prompt - learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            # print(m.isnan().any(), v.isnan().any(), m_hat.isnan().any(), v_hat.isnan().any(), (v_hat < 0).any())
            # print(learning_rate, m_hat.abs().max(), torch.sqrt(v_hat).isnan().any(), torch.sqrt(v_hat).max()+epsilon)
            # validation
            if (iters + 1) % validate_every == 0:
                validation_loss = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, validation_data, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
                loss3.append(validation_loss)
                print(f"Iteration {i}: loss1 {loss}; loss2 {loss_plus}; Validation loss {validation_loss}")
            else:
                print(f"Iteration {i}: loss1 {loss}; loss2 {loss_plus}")
            iters += 1
    # if path does not exist, create the path
    
    with open(os.path.join(output_dir, f"logs.pt"), "wb") as f:
        losses = {'loss1': loss1.copy(), 'loss2': loss2.copy(), 'loss3': loss3.copy()}
        pickle.dump(losses, f)
    # save the final softprompt
    torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{epochs}.pt"))
    return soft_prompt

def zero_order_adam_2c_concurrent(models, tokenizers, sys_prompt, soft_prompt, training_data, validation_data, batchsize, epochs, learning_rate, weight_decay, variation_scale, n_GPU, output_dir, save_per_epochs=1, validate_every=10):
    soft_prompt_copy = soft_prompt.clone()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iter_per_epoch = len(training_data) // batchsize
    # get batched data
    # batched_data = [training_data[i:i+batchsize] for i in range(0, len(training_data), batchsize)]
    # compute parameters in soft_prompt
    dimension = soft_prompt.numel()
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-6
    m = torch.zeros_like(soft_prompt, dtype=torch.float32)
    v = torch.zeros_like(soft_prompt, dtype=torch.float32)
    loss1, loss2, loss3 = [], [], []
    iters = 0
    for e in range(epochs):
        if e % save_per_epochs == 0 and e != 0:
            torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{e}.pt"))
        training_data = [training_data[i] for i in torch.randperm(len(training_data))]
        for i in range(iter_per_epoch):
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            data_temp = training_data[i*batchsize:(i+1)*batchsize]
            
            # random variation in softprompt as uniform unit ball distribution
            random_directions = torch.randn_like(soft_prompt)
            random_directions = random_directions / torch.norm(random_directions)
            random_variations = random_directions * variation_scale
            
            soft_prompt_minus = soft_prompt - random_variations
            # get loss1
            loss_minus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_minus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            # check if loss is nan
            if loss_minus.isnan():
                continue
            loss1.append(loss_minus)
            # print(f'loss:{loss}')
            
            # get variation sampling
            soft_prompt_plus = soft_prompt + random_variations
            loss_plus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_plus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            if loss_plus.isnan():
                continue
            loss2.append(loss_plus)
            # print(f'loss_plus:{loss_plus}')
            # get loss difference
            loss_diff = (loss_plus - loss_minus) / 2
            # compute zero-order gradient
            gradient = (loss_diff / variation_scale * (dimension)) * random_directions
            # update softprompt
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat = m / (1 - beta1 ** (iters+1))
            v_hat = v / (1 - beta2 ** (iters+1))
            soft_prompt = soft_prompt - learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon) - weight_decay * soft_prompt * learning_rate
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            # print(m.isnan().any(), v.isnan().any(), m_hat.isnan().any(), v_hat.isnan().any(), (v_hat < 0).any())
            # print(learning_rate, m_hat.abs().max(), torch.sqrt(v_hat).isnan().any(), torch.sqrt(v_hat).max()+epsilon)
            # validation
            if (iters + 1) % validate_every == 0:
                validation_loss = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, validation_data, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
                loss3.append(validation_loss)
                print(f"Iteration {i}: loss1 {loss_minus}; loss2 {loss_plus}; Validation loss {validation_loss}")
                sqdim = torch.sqrt(torch.tensor(dimension * 1.0))
                pmp_scale = torch.norm(soft_prompt)**2 / sqdim
                update_scale = torch.norm(m_hat * learning_rate / (torch.sqrt(v_hat) + epsilon)) / sqdim
                print(f"Update scale:{update_scale}; Softprompt scale: {pmp_scale}")
                # check if soft_prompt is different from soft_prompt_copy
                print("Difference: ", torch.norm(soft_prompt - soft_prompt_copy))
                # print avg norm
                print(f"Softprompt norm: {torch.norm(soft_prompt)**2 / sqdim}")
            else:
                print(f"Iteration {i}: loss1 {loss_minus}; loss2 {loss_plus}")
            iters += 1
    # if path does not exist, create the path
    
    with open(os.path.join(output_dir, f"logs.pt"), "wb") as f:
        losses = {'loss1': loss1.copy(), 'loss2': loss2.copy(), 'loss3': loss3.copy()}
        pickle.dump(losses, f)
    # save the final softprompt
    torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{epochs}.pt"))
    return soft_prompt

def zero_order_msgd_concurrent(models, tokenizers, sys_prompt, soft_prompt, training_data, validation_data, batchsize, epochs, learning_rate, beta1, weight_decay, variation_scale, n_GPU, output_dir, save_per_epochs=1, validate_every=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iter_per_epoch = len(training_data) // batchsize
    true_epochs = epochs
    if epochs < 1:
        iter_per_epoch = int(len(training_data)*epochs // batchsize)
        epochs = 1
    # get batched data
    # batched_data = [training_data[i:i+batchsize] for i in range(0, len(training_data), batchsize)]
    # compute parameters in soft_prompt
    dimension = soft_prompt.numel()
    m = torch.zeros_like(soft_prompt, dtype=torch.float32)
    loss1, loss2, loss3 = [], [], []
    iters = 0
    for e in range(epochs):
        if e % save_per_epochs == 0 and e != 0:
            torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{e}.pt"))
        training_data = [training_data[i] for i in torch.randperm(len(training_data))]
        for i in range(iter_per_epoch):
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            data_temp = training_data[i*batchsize:(i+1)*batchsize]
            
            # random variation in softprompt as uniform unit ball distribution
            random_directions = torch.randn_like(soft_prompt)
            random_directions = random_directions / torch.norm(random_directions)
            random_variations = random_directions * variation_scale

            soft_prompt_minus = soft_prompt - random_variations
            # get loss1
            loss_minus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_minus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            # check if loss is nan
            if loss_minus.isnan():
                continue
            loss1.append(loss_minus)
            # print(f'loss:{loss}')
            
            # get variation sampling
            soft_prompt_plus = soft_prompt + random_variations
            loss_plus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_plus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            if loss_plus.isnan():
                continue
            loss2.append(loss_plus)
            # print(f'loss_plus:{loss_plus}')
            # get loss difference
            loss_diff = (loss_plus - loss_minus) / 2
            # compute zero-order gradient
            gradient = (loss_diff / variation_scale * (dimension * learning_rate)) * random_directions
            # update softprompt
            m = beta1 * m + (1 - beta1) * gradient
            m_hat = m / (1 - beta1 ** (iters+1))
            soft_prompt = soft_prompt - m_hat - weight_decay * soft_prompt * learning_rate
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            # print(m.isnan().any(), v.isnan().any(), m_hat.isnan().any(), v_hat.isnan().any(), (v_hat < 0).any())
            # print(learning_rate, m_hat.abs().max(), torch.sqrt(v_hat).isnan().any(), torch.sqrt(v_hat).max()+epsilon)
            # validation
            if (iters + 1) % validate_every == 0:
                validation_loss = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, validation_data, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
                loss3.append(validation_loss)
                print(f"Iteration {i}: loss1 {loss_minus}; loss2 {loss_plus}; Validation loss {validation_loss}")
                sqdim = torch.sqrt(torch.tensor(dimension * 1.0))
                grad_scale = torch.norm(gradient) / sqdim
                pmp_scale = torch.norm(soft_prompt) / sqdim
                update_scale = torch.norm(m_hat) / sqdim
                print(f"Lr*Gradient scale: {grad_scale}; Update scale:{update_scale}; Softprompt scale: {pmp_scale}")
            else:
                print(f"Iteration {i}: loss1 {loss_minus}; loss2 {loss_plus}")
            iters += 1
    # if path does not exist, create the path
    
    with open(os.path.join(output_dir, f"logs.pt"), "wb") as f:
        losses = {'loss1': loss1.copy(), 'loss2': loss2.copy(), 'loss3': loss3.copy()}
        pickle.dump(losses, f)
    # save the final softprompt
    torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{true_epochs}.pt"))
    return soft_prompt

def zero_order_gauss_msgd_concurrent(models, tokenizers, sys_prompt, soft_prompt, training_data, validation_data, batchsize, epochs, learning_rate, beta1, weight_decay, variation_scale, n_GPU, output_dir, save_per_epochs=1, validate_every=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iter_per_epoch = len(training_data) // batchsize
    true_epochs = epochs
    if epochs < 1:
        iter_per_epoch = int(len(training_data)*epochs // batchsize)
        epochs = 1
    # get batched data
    # batched_data = [training_data[i:i+batchsize] for i in range(0, len(training_data), batchsize)]
    # compute parameters in soft_prompt
    dimension = soft_prompt.numel()
    m = torch.zeros_like(soft_prompt, dtype=torch.float32)
    loss1, loss2, loss3 = [], [], []
    iters = 0
    for e in range(epochs):
        if e % save_per_epochs == 0 and e != 0:
            torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{e}.pt"))
        training_data = [training_data[i] for i in torch.randperm(len(training_data))]
        for i in range(iter_per_epoch):
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            data_temp = training_data[i*batchsize:(i+1)*batchsize]
            
            # random variation in softprompt as uniform unit ball distribution
            random_directions = torch.randn_like(soft_prompt)
            random_variations = random_directions * variation_scale

            soft_prompt_minus = soft_prompt - random_variations
            # get loss1
            loss_minus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_minus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            # check if loss is nan
            if loss_minus.isnan():
                continue
            loss1.append(loss_minus)
            # print(f'loss:{loss}')
            
            # get variation sampling
            soft_prompt_plus = soft_prompt + random_variations
            loss_plus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_plus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            if loss_plus.isnan():
                continue
            loss2.append(loss_plus)
            # print(f'loss_plus:{loss_plus}')
            # get loss difference
            loss_diff = (loss_plus - loss_minus) / 2
            # compute zero-order gradient
            gradient = (loss_diff / variation_scale * (learning_rate)) * random_directions
            # update softprompt
            m = beta1 * m + (1 - beta1) * gradient
            m_hat = m / (1 - beta1 ** (iters+1))
            soft_prompt = soft_prompt - m_hat - weight_decay * soft_prompt * learning_rate
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            # print(m.isnan().any(), v.isnan().any(), m_hat.isnan().any(), v_hat.isnan().any(), (v_hat < 0).any())
            # print(learning_rate, m_hat.abs().max(), torch.sqrt(v_hat).isnan().any(), torch.sqrt(v_hat).max()+epsilon)
            # validation
            if (iters + 1) % validate_every == 0:
                validation_loss = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, validation_data, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
                loss3.append(validation_loss)
                print(f"Iteration {i}: loss1 {loss_minus}; loss2 {loss_plus}; Validation loss {validation_loss}")
                sqdim = torch.sqrt(torch.tensor(dimension * 1.0))
                grad_scale = torch.norm(gradient) / sqdim
                pmp_scale = torch.norm(soft_prompt) / sqdim
                update_scale = torch.norm(m_hat) / sqdim
                print(f"Lr*Gradient scale: {grad_scale}; Update scale:{update_scale}; Softprompt scale: {pmp_scale}")
            else:
                print(f"Iteration {i}: loss1 {loss_minus}; loss2 {loss_plus}")
            iters += 1
    # if path does not exist, create the path
    
    with open(os.path.join(output_dir, f"logs.pt"), "wb") as f:
        losses = {'loss1': loss1.copy(), 'loss2': loss2.copy(), 'loss3': loss3.copy()}
        pickle.dump(losses, f)
    # save the final softprompt
    torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{true_epochs}.pt"))
    return soft_prompt

def zero_order_proj_msgd_concurrent(models, tokenizers, sys_prompt, project_matrix, soft_prompt_proj, training_data, validation_data, batchsize, epochs, learning_rate, beta1, weight_decay, variation_scale, n_GPU, output_dir, save_per_epochs=1, validate_every=10):
    if soft_prompt_proj is None:
        soft_prompt_proj = torch.randn(1, 7, 10) / 10
    if project_matrix is None:
        project_matrix = torch.randn(soft_prompt_proj.size(0), soft_prompt_proj.size(2), 4096)
    soft_prompt = torch.matmul(soft_prompt_proj, project_matrix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    iter_per_epoch = len(training_data) // batchsize
    true_epochs = epochs
    if epochs < 1:
        iter_per_epoch = int(len(training_data)*epochs // batchsize)
        epochs = 1
    # get batched data
    # batched_data = [training_data[i:i+batchsize] for i in range(0, len(training_data), batchsize)]
    # compute parameters in soft_prompt
    dimension = soft_prompt_proj.numel()
    m = torch.zeros_like(soft_prompt_proj, dtype=torch.float32)
    loss1, loss2, loss3 = [], [], []
    iters = 0
    for e in range(epochs):
        if e % save_per_epochs == 0 and e != 0:
            torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{e}.pt"))
        training_data = [training_data[i] for i in torch.randperm(len(training_data))]
        for i in range(iter_per_epoch):
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            data_temp = training_data[i*batchsize:(i+1)*batchsize]
            
            # random variation in softprompt as uniform unit ball distribution
            random_directions = torch.randn_like(soft_prompt_proj)
            random_directions = random_directions / torch.norm(random_directions)
            random_variations = random_directions * variation_scale

            soft_prompt_minus = soft_prompt - torch.matmul(random_variations, project_matrix)
            # get loss1
            loss_minus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_minus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            # check if loss is nan
            if loss_minus.isnan():
                continue
            loss1.append(loss_minus)
            # print(f'loss:{loss}')
            
            # get variation sampling
            soft_prompt_plus = soft_prompt + torch.matmul(random_variations, project_matrix)
            loss_plus = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt_plus, data_temp, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
            if loss_plus.isnan():
                continue
            loss2.append(loss_plus)
            # print(f'loss_plus:{loss_plus}')
            # get loss difference
            loss_diff = (loss_plus - loss_minus) / 2
            # compute zero-order gradient
            gradient = (loss_diff / variation_scale * (dimension * learning_rate)) * random_directions
            # update softprompt
            m = beta1 * m + (1 - beta1) * gradient
            m_hat = m / (1 - beta1 ** (iters+1))
            soft_prompt_proj = soft_prompt_proj - m_hat - weight_decay * soft_prompt_proj * learning_rate
            soft_prompt = torch.matmul(soft_prompt_proj, project_matrix)
            # print(soft_prompt.shape, soft_prompt.isnan().any(), soft_prompt.isinf().any(), (soft_prompt < 0).any())
            # print(m.isnan().any(), v.isnan().any(), m_hat.isnan().any(), v_hat.isnan().any(), (v_hat < 0).any())
            # print(learning_rate, m_hat.abs().max(), torch.sqrt(v_hat).isnan().any(), torch.sqrt(v_hat).max()+epsilon)
            # validation
            if (iters + 1) % validate_every == 0:
                validation_loss = compute_loss_concurrent(models, tokenizers, sys_prompt, soft_prompt, validation_data, batch_size=batchsize//n_GPU, n_GPU=n_GPU)
                loss3.append(validation_loss)
                print(f"Iteration {i}: loss1 {loss_minus}; loss2 {loss_plus}; Validation loss {validation_loss}")
                sqdim = torch.sqrt(torch.tensor(dimension * 1.0))
                grad_scale = torch.norm(gradient) / sqdim
                pmp_scale = torch.norm(soft_prompt) / sqdim
                update_scale = torch.norm(m_hat) / sqdim
                print(f"Lr*Gradient scale: {grad_scale}; Update scale:{update_scale}; Softprompt scale: {pmp_scale}")
            else:
                print(f"Iteration {i}: loss1 {loss_minus}; loss2 {loss_plus}")
            iters += 1
    # if path does not exist, create the path
    
    with open(os.path.join(output_dir, f"logs.pt"), "wb") as f:
        losses = {'loss1': loss1.copy(), 'loss2': loss2.copy(), 'loss3': loss3.copy()}
        pickle.dump(losses, f)
    # save the final softprompt
    torch.save(soft_prompt, os.path.join(output_dir, f"softprompt_epoch_{true_epochs}.pt"))
    return soft_prompt

import torch
torch.manual_seed(0)
import random
random.seed(0)

with open('/mnt/data/xue.w/yutong/data/grade_school_math/data/test.jsonl', 'r') as f:
    data_test = f.readlines()
    data_test = [json.loads(d) for d in data_test]

with open('/mnt/data/xue.w/yutong/data/grade_school_math/data/train.jsonl', 'r') as f:
    data_train = f.readlines()
    data_train = [json.loads(d) for d in data_train]


soft_prompt = torch.tensor(soft_prompt_init, dtype=torch.float32)
with open('./results/tzo_msgd_spt_det_lr_1e-05_vs_0.001_mo_0.9_wd_0.01/softprompt_epoch_1.pt', "rb") as f:
    soft_prompt = torch.load(f)
# decomposition of soft_prompt: 1*7*4096 -> 1*7*10, 1*10*4096


# zero_order_softprompt_tuning_twopoints_concurrent(models, tokenizers, sys_prompt, soft_prompt_init, data_train, data_test[:128], batchsize=128, epochs=10, learning_rate=1e-8, variation_scale=1e-3, output_dir="softprompt_tuning_bs_128_lr_1e-8_vs_1e-3", save_per_epochs=1, n_GPU=4)
# zero_order_adam_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, data_train[:320], data_test[50:82], batchsize=32, maxIters=10, learning_rate=1e-7, variation_scale=1e-2, n_GPU=8)
# grid search and save all results (including per-iter loss and final softprompt)
# learning_rates = [1e-5, 1e-6, 1e-7]
# variation_scales = [1e-1, 1e-2, 1e-3]
weight_decays = [1e-2]
learning_rates = [1e-5]
momentums = [0.9]
variation_scales = [1e-3]
epochs = 5
batchsize = 32

for weight_decay in weight_decays:
    for momentum in momentums:
        for learning_rate in learning_rates:
            for variation_scale in variation_scales:
                # print hyper-parameters
                print(f"Learning rate: {learning_rate}; Variation scale: {variation_scale}; Weight decay: {weight_decay}")
                # set random seed
                torch.manual_seed(0)
                random.seed(0)
                # zero_order_softprompt_tuning_twopoints_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt_init, data_train[:320], data_test[50:82], batchsize=32, maxIters=10, learning_rate=1e-7, variation_scale=1e-2)
                zero_order_gauss_msgd_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, training_data=data_train, validation_data=data_test[:256], batchsize=256, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, variation_scale=variation_scale, beta1=momentum, n_GPU=8, output_dir=f'./results/tzo_gauss_msgd_spt_det_lr_{learning_rate}_vs_{variation_scale}_mo_{momentum}_wd_{weight_decay}', save_per_epochs=5)
                # zero_order_proj_msgd_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, torch.eye(7).unsqueeze(0), training_data=data_train, validation_data=data_test[:256], batchsize=256, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, variation_scale=variation_scale, beta1=momentum, n_GPU=8, output_dir=f'./results/tzo_proj_msgd_spt_lr_{learning_rate}_vs_{variation_scale}_mo_{momentum}_wd_{weight_decay}_tokendim_{7}', save_per_epochs=1)
                # zero_order_adam_2c_concurrent(peft_models, tokenizers, sys_prompt, soft_prompt, training_data=data_train, validation_data=data_test[:256], batchsize=256, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, variation_scale=variation_scale, n_GPU=8, output_dir=f'./results/tzo_adam_2c_spt_lr_{learning_rate}_vs_{variation_scale}_wd_{weight_decay}', save_per_epochs=1, validate_every=10)
