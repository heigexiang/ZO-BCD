# import dependencies
import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model_path = "/mnt/xue.w/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)



data = load_dataset("/mnt/yutong/data/grade_school_math/data")
data_train, data_test, data_val = data["train"], data["test"], data["validation"]

def generate_prompt(question, answer=None, eos_token="</s>"):
    # instruction = "Solving the follwing math problem and response with '\n#### <answer>' with <answer> substituted by the correct number in the very end:\n"
    input = "<|system|>:" + "Solving the following math problem and respond with '\n#### <answer>' with <answer> substituted by the correct number in the very end:\n" + "</s>" + "\n<|user|>:" + question + "</s>\n<|assistant|>:"
    # input = f"{question}\n"
    answer = f"{answer + ' ' + eos_token if answer else ''} "
    # answer = f"Answer: {answer + ' ' + eos_token if answer else ''} "
    prompt = (" ").join([input, answer])
    # prompt = (" ").join([instruction, input, answer])
    return prompt

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)

from peft import PeftModel

peft_model_id = "llama_2_7b_lora_2/1_epoch_finetuning"
peft_models = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16)


output_dir = "llama_2_7b_lora_3"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
per_device_eval_batch_size = 4
eval_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 50
learning_rate = 5e-4
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
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,

    # max_steps=max_steps,
    num_train_epochs=5,
    
    warmup_ratio=warmup_ratio,
    evaluation_strategy=evaluation_strategy,
    lr_scheduler_type=lr_scheduler_type,
    group_by_length=True,
    ddp_find_unused_parameters=False,
)

def formatting_func(prompt):
    output = []

    for d, s in zip(prompt["question"], prompt["answer"]):
        op = generate_prompt(d, s)
        output.append(op)

    return output

# response_template = "<|assistant|>:"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# only use the first half of data_train
data_train = data_train.select(range(len(data_train)//2))
data_val = data_val.select(range(len(data_val)//2))

trainer = SFTTrainer(
    model=model,
    train_dataset=data_train,
    eval_dataset=data_val,
    peft_config=lora_config,
    formatting_func=formatting_func,
    # data_collator=collator,
    # max_seq_length=1024,
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_args,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.save_model(f"{output_dir}/5_epoch_finetuning")