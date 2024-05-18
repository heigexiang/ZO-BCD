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
# mp.set_start_method('spawn')
import json, pickle, copy
from peft import PeftModel
from tqdm import tqdm
import re
import concurrent.futures

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

model_pth = '../Meta-Llama-3-8B-Instruct'
# peft_model_pth = './llama_2_7b_lora_3/5_epoch_finetuning'

tokenizer = AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)
model = AutoModelForCausalLM.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)

with open('../data/grade_school_math/data/test.jsonl', 'r') as f:
    data_test = f.readlines()
    data_test = [json.loads(d) for d in data_test]

with open('../data/grade_school_math/data/train.jsonl', 'r') as f:
    data_train = f.readlines()
    data_train = [json.loads(d) for d in data_train]

with open('./generated_data/actor_response_data_01.jsonl', 'r') as f:
    actor_response_data = f.readlines()
    actor_response_data = [json.loads(d) for d in actor_response_data]

models = [copy.deepcopy(model).to(torch.device(f'cuda:{_}')) for _ in range(8)]
tokenizers = [AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False) for _ in range(8)]
for tokenizer in tokenizers:
    tokenizer.pad_token = tokenizer.bos_token
critic_models = models
# actor_models = [PeftModel.from_pretrained(models[i], peft_model_pth, torch_dtype=torch.float16) for i in range(8)]
embedding_dim = 4096

# id = 0
# for data in data_train + data_test:
#     data['id'] = id
#     id += 1

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return "[invalid]"    

def extract_judgement(output):
    if output.startswith("#### The answer is: Accepted."):
        return 'Accepted'
    if output.startswith("#### The answer is: Wrong Answer."):
        return 'Wrong Answer'
    return 'Invalid'

def generate_critic_response(critic, tokenizer, actor_data, question_data, batch_size, device, use_tqdm=False):
    n_data = len(actor_data)
    n_batches = n_data // batch_size
    results = []
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
        pbar = tqdm(total=n_data, desc=f'Data generating end to end on {device}', ncols=100)
    for batch_idx in range(n_batches):
        batch_data = actor_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
        questions = [question_data[d['question_id']]['question'] for d in batch_data]
        answers = [extract_answer(question_data[d['question_id']]['answer']) for d in batch_data]
        ids = [d['id'] for d in batch_data]
        actor_responses = [d['actor_response'] for d in batch_data]
        correct_judges = ['Accepted' if d['answer_correct'] == d['answer_actor'] else 'Wrong Answer' for d in batch_data]

        # generate batched input embeddings, attention mask for critic and apply to generate_kwargs
        messages = [[{'role': 'system', 'content': 'You are a critic who is responsible for judging the correctness of the actor\'s answer. Provided with the math problem, correct answer and the student\'s answer, you need to judge whether the actor\'s answer is correct. If the actor\'s answer is right, respond with "#### The answer is: Accepted." Otherwise, analyze the reason why the actor arrived at the wrong answer and respond with "#### The answer is: Wrong Answer. [Reason for the wrong answer, without displaying the correct number to the question]".'},
                     {'role': 'question', 'content': question},
                     {'role': 'correct answer', 'content': answer},
                     {'role': 'actor\'s answer', 'content': actor_response}] for question, answer, actor_response in zip(questions, answers, actor_responses)]
        # input_prompts = ['<|system|>:' + sys_prompt + '</s>\n<|question|>:' + question + '</s>\n<|correct answer|>:'+ answer + '</s>\n<|student|>:' + actor_response + '</s>\n<|assistant|>:' for question, answer, actor_response in zip(questions, answers, actor_responses)]
        input_prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        input_embeds = [critic.get_input_embeddings()(tokenizer(input_prompt, return_tensors='pt', padding=False, truncation=True, max_length=1024).input_ids.to(device)) for input_prompt in input_prompts]
        max_len = max([input_embed.size(1) for input_embed in input_embeds])
        attention_mask = torch.concatenate([torch.cat([torch.zeros(max_len - input_embed.size(1), device=device), torch.ones(input_embed.size(1), device=device)]).unsqueeze(0) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        input_embeds = torch.concatenate([torch.cat([torch.zeros(1, max_len - input_embed.size(1), embedding_dim, device=device), input_embed], dim=1) for input_embed in input_embeds], dim=0).to(dtype=torch.float16)
        generate_kwargs['inputs_embeds'] = input_embeds
        generate_kwargs['attention_mask'] = attention_mask

        # generate critic responses
        outputs = critic.generate(**generate_kwargs)
        for critic_output in outputs:
            if tokenizer.eos_token_id in critic_output:
                eos_idx = (critic_output == tokenizer.eos_token_id).nonzero()[0].item()
                critic_output[eos_idx+1:] = tokenizer.pad_token_id
        critic_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        results.extend([{'actor_response_id': id, 'critic_response': critic_response, 'judge_correct': correct_judge, 'judge_critic': extract_judgement(critic_response)} for id, critic_response, answer, actor_response, correct_judge in zip(ids, critic_responses, answers, actor_responses, correct_judges)])
        if use_tqdm:
            pbar.update(batch_size)
    if use_tqdm:
        pbar.close()
    return results

def generate_critic_response_data(critics, tokenizers, actor_data, question_data, batch_size, n_GPU=8, use_tqdm=False, save_pth='./generated_data/actor_response_data.jsonl', start_id=None):

    n_data = len(actor_data)
    data_per_GPU = n_data // n_GPU
    batchsize_per_GPU = batch_size // n_GPU
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_GPU):
            futures.append(executor.submit(generate_critic_response, critics[i], tokenizers[i], actor_data[i*data_per_GPU:(i+1)*data_per_GPU], data_train, batchsize_per_GPU, torch.device(f'cuda:{i}'), use_tqdm=(use_tqdm and i == 0)))
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    # check if save path exists
    save_dir = os.path.dirname(save_pth)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # check if save file exists
    if os.path.exists(save_pth):
        with open(save_pth, 'r') as f:
            saved_data = f.readlines()
            saved_data = [json.loads(d) for d in saved_data]
        # check the largest id in saved_data
        if start_id is None:
            start_id = max([d['id'] for d in saved_data]) + 1
    else:
        if start_id is None:
            start_id = 0
    # add id to results
    for i, result in enumerate(results):
        result['id'] = start_id + i
    # save results to file
    with open(save_pth, 'a') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    return results   

# actor_sys_prompt = "Solving the following math problem and respond with '\n#### <answer>' with <answer> substituted by the correct number in the very end:\n"

epochs = 1
for e in range(epochs):
    generate_critic_response_data(critic_models, tokenizers, actor_response_data[70144:], data_train, 64, n_GPU=8, use_tqdm=True, save_pth=f'./generated_data/critic_response_data_02.jsonl', start_id=None)
    print(f'Epoch {e} done')
    # check number of lines in the file
    with open(f'./generated_data/critic_response_data_02.jsonl', 'r') as f:
        print(f'Number of lines in the file: {len(f.readlines())}')
