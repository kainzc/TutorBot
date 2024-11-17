

# load trained model
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import os
import torch
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


model_id = "google/gemma-2b"
model_folder = "./outputs/finetuned_model"
new_finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
                        model_folder,
                        low_cpu_mem_usage=True,
                        return_dict = True,
                        torch_dtype = torch.float16,
                        device_map = "cuda:0",)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
tokenizer.padding_side = "left" # padding side is left
text = "Quote: Imagination is"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = new_finetuned_model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

