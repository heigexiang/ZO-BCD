import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_pth = '../Llama-2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)
tokenizer = AutoTokenizer.from_pretrained(model_pth,torch_dtype=torch.float16,load_in_4bit=False,load_in_8bit=False)
spt0 = model.get_input_embeddings()(tokenizer("1+1=?", return_tensors="pt").input_ids)
torch.save(spt0, 'softprompt_init.pt')