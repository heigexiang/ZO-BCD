from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

accelerator = Accelerator()

def write_pretty_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

with open('/mnt/yutong/data/grade_school_math/data/train.jsonl', 'r') as f:
    data_train = f.readlines()
    data_train = [json.loads(d) for d in data_train]

with open('/mnt/yutong/data/grade_school_math/data/test.jsonl', 'r') as f:
    data_test = f.readlines()
    data_test = [json.loads(d) for d in data_test]

tiny_llama = "/mnt/xue.w/models/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/77e23968eed12d195bd46c519aa679cc22a27ddc"
llama_7b_hf_chat = "/mnt/xue.w/models/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
# code_llama_70 = '/mnt/xue.w/models/hub/models--codellama--CodeLlama-70b-hf/snapshots/4570a4edc524fb9f20f605b417bb43828fa5997a'

miqu_70b ='/mnt/xue.w/models/hub/models--miqudev--miqu-1-70b/models--miqudev--miqu-1-70b/snapshots/82f0daa6767263aa5990dea54dbb13e94d096de7'
Mixtral_8x7b_instruct ='/mnt/xue.w/models/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/125c431e2ff41a156b9f9076f744d2f35dd6e67a'

model_pth = llama_7b_hf_chat
model = AutoModelForCausalLM.from_pretrained(
    model_pth,
    device_map={"": accelerator.process_index},
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_pth)
tokenizer.pad_token = tokenizer.eos_token

from peft import PeftModel

peft_model_id = "llama_2_7b_lora_2/1_epoch_finetuning"
peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16)

from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return True
        return False

# softprompt training in gsm8k dataset
import re
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

def compute_loss(model, sys_prompt, soft_prompt, data):

    batch_size = len(data)
    total_loss = 0
    accelerator.wait_for_everyone()
    embeddings_all = [torch.cat([soft_prompt, model.get_input_embeddings()(tokenizer("<|system|>:" + sys_prompt + "</s>" + "\n<|user|>:" + datum['question'] + "</s>\n<|assistant|>:", return_tensors="pt").input_ids.to(device))], dim=1) for datum in data]
    true_ans_all = [float(extract_answer(datum['answer'])) for datum in data]
    data_pair = [(embeddings, true_ans) for embeddings, true_ans in zip(embeddings_all, true_ans_all)]
    with accelerator.split_between_processes(data_pair) as split_data:
        for i, (inputs_embeds, true_ans) in enumerate(split_data):
            streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
            stop = StopOnTokens()
            generate_kwargs = dict(
                inputs_embeds=inputs_embeds,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                num_beams=1,
                stopping_criteria=StoppingCriteriaList([stop])
            )
            model.generate(**generate_kwargs)
            output = ""
            for new_token in streamer:
                output += new_token
                if '</s>' in output:
                    break
            ans = extract_answer(output)
            if ans == "[invalid]":
                loss = torch.tensor(1.0)
            else:
                ans = float(ans)
                if ans == 0 and true_ans == 0:
                    loss = 0
                else:
                    loss = torch.abs(torch.tensor(ans - true_ans)) / (torch.abs(torch.tensor(true_ans)) + torch.abs(torch.tensor(ans)))
            total_loss += loss
        total_loss = [total_loss]

    loss_gathered = gather_object(total_loss)
    loss = sum(loss_gathered) / batch_size
    return loss
    
def zero_order_softprompt_tuning_twopoints(model, sys_prompt, soft_prompt, training_data, validation_data, batchsize, maxIters, learning_rate, variation_scale):
    # get batched data
    batched_data = [training_data[i:i+batchsize] for i in range(0, len(training_data), batchsize)]
    # compute parameters in soft_prompt
    dimension = soft_prompt.numel()
    for i in range(maxIters):
        # get initial loss
        loss = compute_loss(model, sys_prompt, soft_prompt, batched_data[i])
        # random variation in softprompt as uniform unit ball distribution
        # if accelerator.is_main_process:
        random_directions = torch.randn_like(soft_prompt)
        random_directions = random_directions / torch.norm(random_directions)
        random_variations = random_directions * variation_scale
        # get variation sampling
        soft_prompt_plus = soft_prompt + random_variations
        loss_plus = compute_loss(model, sys_prompt, soft_prompt_plus, batched_data[i])
        # get loss difference
        loss_diff = loss_plus - loss
        # compute zero-order gradient
        gradient = (loss_diff / variation_scale * dimension) * random_directions
        # update softprompt
        soft_prompt = soft_prompt - learning_rate * gradient
        # validation
        validation_loss = compute_loss(model, sys_prompt, soft_prompt, validation_data)
        if accelerator.is_main_process:
            print(f"Iteration {i}: Validation loss {validation_loss}")
    return soft_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_tokens = 10
sys_prompt = "Solving the following math problem and respond with '\n#### <answer>' with <answer> substituted by the correct number in the very end:\n"
soft_prompt = torch.randn(1, n_tokens, embedding_dim).to(device).to(torch.float16)
zero_order_softprompt_tuning_twopoints(peft_model, sys_prompt, soft_prompt, data_test[:80], data_test[50:54], batchsize=4, maxIters=20, learning_rate=1e-7, variation_scale=1e-2)