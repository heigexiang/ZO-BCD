# import dependencies
import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_dataset
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig
)
from torch.utils.data import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import pickle
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import PartialState
device_string = PartialState().process_index

# parser = argparse.ArgumentParser()
# parser.add_argument("--local-rank", type=int)
# args = parser.parse_args()
model_path = '../Meta-Llama-3-8B-Instruct'
# device_map={'':torch.cuda.current_device()}
device_map = {'':device_string}
# dist.init_process_group(backend='nccl')
# rank = dist.get_rank()
# device = torch.device("cuda", rank)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(model_path)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# model.to(torch.device("cuda"))
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)

# data = load_dataset("/mnt/yutong/data/grade_school_math/data")
# data_train, data_test, data_val = data["train"], data["test"], data["validation"]

# def generate_prompt(question, answer=None, eos_token="</s>"):
#     # instruction = "Solving the follwing math problem and response with '\n#### <answer>' with <answer> substituted by the correct number in the very end:\n"
#     input = "<|system|>:" + "Solving the following math problem and respond with '\n#### <answer>' with <answer> substituted by the correct number in the very end:\n" + "</s>" + "\n<|user|>:" + question + "</s>\n<|assistant|>:"
#     # input = f"{question}\n"
#     answer = f"{answer + ' ' + eos_token if answer else ''} "
#     # answer = f"Answer: {answer + ' ' + eos_token if answer else ''} "
#     prompt = (" ").join([input, answer])
#     # prompt = (" ").join([instruction, input, answer])
#     return prompt


# tokenizer.add_special_tokens({"pad_token": "<PAD>"})
tokenizer.pad_token = tokenizer.bos_token
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)




# model = model.to(device)
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])


from peft import PeftModel

# peft_model_id = "llama_2_7b_lora_2/1_epoch_finetuning"
# peft_models = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer):
        super(SupervisedDataset, self).__init__()

        # rank0_print("Formatting inputs...")
        # sources = [example["conversations"] for example in raw_data]
        # self.template = template
        # data_dict = preprocess(sources, tokenizer, self.template)
        self.input_ids = raw_data["input_ids"]
        self.labels = raw_data["labels"]
        self.attention_mask = raw_data["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
    
with open('./generated_data/processed_actor_positive_data.pkl', 'rb') as f:
    complete_dataset = pickle.load(f)
n_training_data = len(complete_dataset['input_ids']) - 128
training_dataset = dict(
    input_ids=complete_dataset['input_ids'][:n_training_data],
    labels=complete_dataset['labels'][:n_training_data],
    attention_mask=complete_dataset['attention_mask'][:n_training_data]
)
validation_dataset = dict(
    input_ids=complete_dataset['input_ids'][n_training_data:],
    labels=complete_dataset['labels'][n_training_data:],
    attention_mask=complete_dataset['attention_mask'][n_training_data:]
)
data_train = SupervisedDataset(training_dataset, tokenizer)
data_val = SupervisedDataset(validation_dataset, tokenizer)




# torch.cuda.set_device(args.local_rank)
# dist.init_process_group("gloo", rank=args.local_rank, world_size=8)
# model = model.to(args.local_rank)
# ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

# train_sampler = DistributedSampler(data_train)
# val_sampler = DistributedSampler(data_val)

# train_loader = DataLoader(data_train, batch_size=32, sampler=train_sampler)
# val_loader = DataLoader(data_val, batch_size=32, sampler=val_sampler)

output_dir = "finetuned_models/actor_llama3_lora/fo_lr_0.0001_fp8"
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
per_device_eval_batch_size = 4
eval_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_strategy = "epoch"
# save_steps = 500
logging_steps = 50
learning_rate = 1e-4
max_grad_norm = 0.3
max_steps = 50
warmup_ratio = 0.03
evaluation_strategy = "steps"
lr_scheduler_type = "constant"

training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,
    eval_accumulation_steps=eval_accumulation_steps,
    optim=optim,
    save_strategy=save_strategy,
    # save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,

    # max_steps=max_steps,
    num_train_epochs=1,
    # strategy="distributed",
    
    warmup_ratio=warmup_ratio,
    evaluation_strategy=evaluation_strategy,
    lr_scheduler_type=lr_scheduler_type,
    group_by_length=True,
    ddp_find_unused_parameters=False,
)

# def formatting_func(prompt):
#     output = []

#     for d, s in zip(prompt["question"], prompt["answer"]):
#         op = generate_prompt(d, s)
#         output.append(op)

#     return output

# response_template = "<|assistant|>:"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# only use the first half of data_train
# data_train = data_train.select(range(len(data_train)//2))
# data_val = data_val.select(range(len(data_val)//2))

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=data_train,
#     eval_dataset=data_val,
#     peft_config=lora_config,
#     formatting_func=formatting_func,
#     # data_collator=collator,
#     # max_seq_length=1024,
#     max_seq_length=4096,
#     tokenizer=tokenizer,
#     args=training_args,
# )
trainer = Trainer(
    # model=ddp_model,
    model=model,
    train_dataset=data_train,
    eval_dataset=data_val,
    # train_dataloader=train_loader,
    # eval_dataloader=val_loader,
    tokenizer=tokenizer,
    args=training_args
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.save_model(f"{output_dir}/1_epoch_finetuning")