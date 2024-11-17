import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from pathlib import Path
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv, find_dotenv
from peft import LoraConfig  # Import LoraConfig

load_dotenv(find_dotenv(filename=".env.local"))

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Load data
data_folder = Path("./test_data")
csv_files = list(data_folder.glob("*.csv"))
print(f"CSV files found: {csv_files}")

data_frames = [pd.read_csv(file) for file in csv_files]
combined_data = pd.concat(data_frames, ignore_index=True)
print("Combined Data:")
print(combined_data.head())

hf_dataset = Dataset.from_pandas(combined_data)
print("HF Dataset (initial):")
print(hf_dataset[:5])

# Initialize tokenizer
model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get('HF_TOKEN', ''))
tokenizer.padding_side = "right"  # Set padding side to 'right' as per warning

# Tokenize the dataset
def tokenize_function(example):
    text = f"Question: {example['Question']}\nAnswer: {example['Answer']}<eos>"
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": tokens["input_ids"][0],
        "attention_mask": tokens["attention_mask"][0]
    }

hf_dataset = hf_dataset.map(tokenize_function, batched=False)
print("Tokenized HF Dataset:")
print(hf_dataset[:5])

# Initialize LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Initialize model with quantization and device mapping
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Ensure the output folder exists
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

# Define the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=str(output_folder),
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config  
)

print("Starting training process...")
trainer.train()

fine_tuned_model_folder = output_folder / "finetuned_model"
fine_tuned_model_folder.mkdir(exist_ok=True)
trainer.save_model(str(fine_tuned_model_folder))
print(f"Model saved to {fine_tuned_model_folder}")
