print('finetuning summarizer with lr=1e-3, zo_eps=1e-3, fp16, epoch2')


# import libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
import json, pickle, copy
from peft import PeftModel
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig
)
from tqdm import tqdm
import re
import concurrent.futures
from transformers.trainer_pt_utils import LabelSmoother

# Load LLaMA3 model
model_pth = '../Meta-Llama-3-8B-Instruct'


model = AutoModelForCausalLM.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)

# model = AutoModelForCausalLM.from_pretrained(model_pth,torch_dtype=torch.float32,load_in_4bit=False,load_in_8bit=False)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

models = [copy.deepcopy(model).to(torch.device(f'cuda:{_}')) for _ in range(8)]
# tokenizers = [AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False) for _ in range(8)]

for model in models:
    model.trainable_parameters = [(name, parameters) for name, parameters in model.named_parameters() if parameters.requires_grad]

for name, module in model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)
        # print(f'{name} is converted to float32')

# for tokenizer in tokenizers:
#     tokenizer.pad_token = tokenizer.bos_token

# with open('./generated_data/processed_actor_positive_data.pkl', 'rb') as f:
# with open('./generated_data/processed_critic_positive_data.pkl', 'rb') as f:
with open('./generated_data/processed_summarizer_positive_data.pkl', 'rb') as f:
    complete_dataset = pickle.load(f)
item_counts = [complete_dataset['labels'][id].ne(-100).sum().item() for id in range(complete_dataset['labels'].size(0))]
complete_dataset['item_counts'] = item_counts

n_training_data = len(complete_dataset['input_ids']) - 128
training_dataset = dict(
    input_ids=complete_dataset['input_ids'][:n_training_data],
    labels=complete_dataset['labels'][:n_training_data],
    attention_mask=complete_dataset['attention_mask'][:n_training_data],
    item_counts=complete_dataset['item_counts'][:n_training_data]
)
validation_dataset = dict(
    input_ids=complete_dataset['input_ids'][n_training_data:],
    labels=complete_dataset['labels'][n_training_data:],
    attention_mask=complete_dataset['attention_mask'][n_training_data:],
    item_counts=complete_dataset['item_counts'][n_training_data:]
)

def evaluate_single_node(model, data_val, scale_factor, device):
    with torch.no_grad():
        return model(input_ids=data_val['input_ids'].to(device),
                    labels=data_val['labels'].to(device),
                    attention_mask=data_val['attention_mask'].to(device)).loss.item() * scale_factor

def evaluate_multi_node(models, data_val, n_GPU):

    n_data = len(data_val['input_ids'])
    n_data_per_GPU = n_data // n_GPU

    item_counts = data_val['item_counts']

    scale_factors = [sum(item_counts[i * n_data_per_GPU: (i+1) * n_data_per_GPU]) for i in range(n_GPU)]
    loss = 0.0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_GPU):
            futures.append(executor.submit(evaluate_single_node, models[i], 
                                           dict(input_ids=data_val['input_ids'][i * n_data_per_GPU: (i+1) * n_data_per_GPU],
                                                labels=data_val['labels'][i * n_data_per_GPU: (i+1) * n_data_per_GPU],
                                                attention_mask=data_val['attention_mask'][i * n_data_per_GPU: (i+1) * n_data_per_GPU]),
                                           scale_factors[i], torch.device(f'cuda:{i}')))
        for future in concurrent.futures.as_completed(futures):
            loss_delta = future.result()
            loss += loss_delta
    
    total_scale_factor = sum(scale_factors)
    return loss / total_scale_factor 


def compute_loss_single_node(model, input_ids, labels, attention_mask, zo_eps, random_seed, batch_size, scale_factor, device, reset=False):

    n_data = len(input_ids)
    n_batch = n_data // batch_size

    # random perturbation
    generator = torch.Generator().manual_seed(random_seed)
    for _, params in model.trainable_parameters:
        z = torch.normal(mean=0, std=1, size=params.data.size(), generator=generator, dtype=params.data.dtype).to(device=device)
        params.data += z * zo_eps
    
    # compute loss1
    loss1 = 0
    with torch.no_grad():
        for batch_id in range(n_batch):
            loss1 += model(input_ids=input_ids[batch_id * batch_size: (batch_id + 1) * batch_size].to(device),
                           labels=labels[batch_id * batch_size: (batch_id + 1) * batch_size].to(device),
                           attention_mask=attention_mask[batch_id * batch_size: (batch_id + 1) * batch_size].to(device)).loss.item()
    
    # perturbate in the opposite direction
    generator = generator.manual_seed(random_seed)
    for _, params in model.trainable_parameters:
        z = torch.normal(mean=0, std=1, size=params.data.size(), generator=generator, dtype=params.data.dtype).to(device=device)
        params.data -= z * zo_eps * 2
    
    # compute loss2
    loss2 = 0
    with torch.no_grad():
        for batch_id in range(n_batch):
            loss2 += model(input_ids=input_ids[batch_id * batch_size: (batch_id + 1) * batch_size].to(device),
                           labels=labels[batch_id * batch_size: (batch_id + 1) * batch_size].to(device),
                           attention_mask=attention_mask[batch_id * batch_size: (batch_id + 1) * batch_size].to(device)).loss.item()
    
    if reset:
        generator = generator.manual_seed(random_seed)
        for _, params in model.trainable_parameters:
            z = torch.normal(mean=0, std=1, size=params.data.size(), generator=generator, dtype=params.data.dtype).to(device=device)
            params.data += z * zo_eps
    
    return loss1 * scale_factor, loss2 * scale_factor # the scale_factor is corresponding to the number of minibatches in the computation of the CE loss (i.e., number of non-[-100] labels)

def compute_loss_multi_node(models, input_ids, labels, attention_mask, zo_eps, random_seed, batch_size, item_counts=None, n_GPU=8, reset=False): # batch_size is exactly the batch_size per node

    n_data = len(input_ids)
    n_data_per_GPU = n_data // n_GPU

    if item_counts is None:
        item_counts = [labels[id].ne(-100).sum().item() for id in range(labels.size(0))]
    
    scale_factors = [sum(item_counts[i * n_data_per_GPU: (i+1) * n_data_per_GPU]) for i in range(n_GPU)]
    loss1, loss2 = 0.0, 0.0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_GPU):
            futures.append(executor.submit(compute_loss_single_node, models[i], 
                                           input_ids[i * n_data_per_GPU: (i+1) * n_data_per_GPU], 
                                           labels[i * n_data_per_GPU: (i+1) * n_data_per_GPU], 
                                           attention_mask[i * n_data_per_GPU: (i+1) * n_data_per_GPU],
                                           zo_eps, random_seed, batch_size, scale_factors[i],
                                           torch.device(f'cuda:{i}'), reset))
        for future in concurrent.futures.as_completed(futures):
            loss1_delta, loss2_delta = future.result()
            loss1 += loss1_delta
            loss2 += loss2_delta
    
    total_scale_factor = sum(scale_factors)
    return loss1 / total_scale_factor, loss2 / total_scale_factor   

def update_params_single_node(model, zo_eps, random_seed, scale_factor, device, reset=False): # scale factor here denotes the gradient step
    
    generator = torch.Generator().manual_seed(random_seed)
    for _, params in model.trainable_parameters:
        z = torch.normal(mean=0, std=1, size=params.data.size(), generator=generator, dtype=params.data.dtype).to(device=device)
        params.data += z * (zo_eps - scale_factor) if not reset else z * (-scale_factor)

    return

def update_params_multi_node(models, zo_eps, random_seed, scale_factor, n_GPU=8, reset=False):

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_GPU):
            futures.append(executor.submit(update_params_single_node, models[i], zo_eps, random_seed, scale_factor, torch.device(f'cuda:{i}'), reset))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    return

def zo_lora_finetuning_multi_node(models, data_train, data_val, zo_eps, learning_rate, batch_size, random_seeds=None, n_iters=None, n_epochs=None, n_GPU=8, verbose=False, validate_every=20):
    
    assert (n_iters is None) ^ (n_epochs is None)
    input_ids, labels, attention_mask, item_counts = data_train['input_ids'], data_train['labels'], data_train['attention_mask'], data_train['item_counts']
    
    n_iters_per_epoch = len(input_ids) // batch_size
    n_iters = n_iters_per_epoch * n_epochs if n_epochs is not None else n_iters
    batch_size_per_GPU = batch_size // n_GPU

    if random_seeds is None:
        random_seeds = torch.randint(100000000, (n_iters,))

    e = 0 # epoch
    iter = 0 # iteration
    loss1_hist, loss2_hist, evaluation_hist = [], [], []
    while iter < n_iters:

        # shuffle data for epoch e
        shuffled_ids = torch.randperm(len(input_ids))

        for i in range(n_iters_per_epoch):

            if iter >= n_iters:
                break

            # get iter data
            batch_input_ids = input_ids[shuffled_ids[i * batch_size: (i+1) * batch_size]]
            batch_labels = labels[shuffled_ids[i * batch_size: (i+1) * batch_size]]
            batch_attention_mask = attention_mask[shuffled_ids[i * batch_size: (i+1) * batch_size]]
            batch_item_counts = [item_counts[shuffled_ids[j]] for j in range(i * batch_size, (i+1) * batch_size)]

            # compute loss1, loss2
            loss1, loss2 = compute_loss_multi_node(models=models, 
                                                   input_ids=batch_input_ids, 
                                                   labels=batch_labels, 
                                                   attention_mask=batch_attention_mask, 
                                                   zo_eps=zo_eps, 
                                                   random_seed=random_seeds[iter].item(),
                                                   batch_size=batch_size_per_GPU,
                                                   item_counts=batch_item_counts,
                                                   n_GPU=n_GPU,
                                                   reset=False)
            
            # compute update stepsize
            scale_factor = (loss1 - loss2) / (2 * zo_eps) * learning_rate

            # update params
            update_params_multi_node(models=models, 
                                     zo_eps=zo_eps, 
                                     random_seed=random_seeds[iter].item(),
                                     scale_factor=scale_factor,
                                     n_GPU=n_GPU,
                                     reset=False)

            if verbose:
                print(f'In epoch {e} iter {i}, loss1={loss1}, loss2={loss2}')
            iter += 1
            loss1_hist.append(loss1)
            loss2_hist.append(loss2)
            if iter % validate_every == 0:
                loss_val = evaluate_multi_node(models=models, 
                                                 data_val=data_val, 
                                                 n_GPU=n_GPU)
                if verbose:
                    print(f'Validation loss={loss_val}')
                evaluation_hist.append(loss_val)
        
        e += 1

    return loss1_hist, loss2_hist, evaluation_hist

if __name__ == '__main__':

    zo_eps = 1e-3
    learning_rate = 1e-3
    # learning_rate = 0 # for debugging

    loss1_hist, loss2_hist, evaluation_hist = zo_lora_finetuning_multi_node(models=models, 
                                                                            data_train=training_dataset, 
                                                                            data_val=validation_dataset,
                                                                            zo_eps=zo_eps,
                                                                            learning_rate=learning_rate,
                                                                            batch_size=64,
                                                                            random_seeds=None,
                                                                            n_iters=None,
                                                                            n_epochs=2,
                                                                            n_GPU=8,
                                                                            verbose=True,
                                                                            validate_every=20)

    # # save hist
    dir = f'./finetuned_models/summarizer_llama3_lora/zo_eps_{zo_eps}_lr_{learning_rate}_fp16_epoch_2'
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + '/logs.pkl', 'wb') as f:
        pickle.dump(dict(loss1=loss1_hist, loss2=loss2_hist, val_loss=evaluation_hist), f)
    print('Logs saved.')
    param_dict = dict()
    for name, params in model.trainable_parameters:
        param_dict[name] = params.data.cpu().numpy()
    with open(dir + '/params.pkl', 'wb') as f:
        pickle.dump(param_dict, f)
    print('Params saved.')