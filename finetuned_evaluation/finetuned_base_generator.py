import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from dotenv import load_dotenv, find_dotenv
import json
import requests
import time
import re  

load_dotenv(find_dotenv(filename=".env.local"))

api_key = os.environ.get('OPENAI_API_KEY')
HF_TOKEN = os.environ.get('HF_TOKEN', '')

test_set_folder = './test_set'
output_jsonl = 'output.jsonl'


def generate_answer(model, tokenizer, question, device):
    input_text = f"Question: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)

    model.eval()
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    if 'Answer:' in generated_text:
        answer = generated_text.split('Answer:')[1].strip()
    else:
        answer = generated_text.strip()
    return answer


def is_response_complete(evaluation):
    required_keys = ['base_score', 'base_reasoning', 'fine_tuned_score', 'fine_tuned_reasoning']
    return all(key in evaluation and evaluation[key] for key in required_keys)


def clean_json_response(response_text):
    # Remove triple backticks and any language specifier
    cleaned_text = re.sub(r"```(json)?", "", response_text).strip()
    return cleaned_text


def send_to_api(prompt):
    time.sleep(1)  
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
        {
            "role": "system",
            "content": "You are a teaching assistant designed to output JSON."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 2000,     
        "temperature": 0.4,    
        "top_p": 0.9,            
        "frequency_penalty": 0.5,
        "presence_penalty": 0.0, 
    }

    print("Sending API request for grading...")
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        response_data = response.json()
        assistant_message = response_data['choices'][0]['message']['content']
        print("API response received.")
        print(f"Assistant's raw response:\n{assistant_message}\n")
        
        assistant_message = clean_json_response(assistant_message)
        
        return assistant_message
    else:
        print(f"Error response: {response.text}")
        return None


def process_test_set():
    model_id = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN
    ).to(device)

    fine_tuned_model_folder = "outputs/finetuned_model"
    if not os.path.exists(fine_tuned_model_folder):
        raise FileNotFoundError(f"Fine-tuned model folder not found at {fine_tuned_model_folder}")

    print("Loading fine-tuned model...")
    fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_folder).to(device)

    csv_files = [f for f in os.listdir(test_set_folder) if f.endswith('.csv')]
    total_files = len(csv_files)
    for file_index, csv_file in enumerate(csv_files, start=1):
        file_path = os.path.join(test_set_folder, csv_file)
        print(f"\nProcessing file ({file_index}/{total_files}): {csv_file}")
        df = pd.read_csv(file_path)

        total_questions = len(df)
        for index, row in df.iterrows():
            file_name = row.get('File Name', '')
            page_number = row.get('Page Number', '')
            question = row.get('Question', '')
            reference_answer = row.get('Answer', '')
            subject = row.get('Subject', '')

            if not question:
                continue 

            print(f"\nProcessing question {index + 1}/{total_questions}: {question}")

            # Generate answers from both models
            print("Generating answer from base model...")
            base_answer = generate_answer(base_model, tokenizer, question, device)
            print(f"Base Model Answer:\n{base_answer}\n")

            print("Generating answer from fine-tuned model...")
            fine_tuned_answer = generate_answer(fine_tuned_model, tokenizer, question, device)
            print(f"Fine-Tuned Model Answer:\n{fine_tuned_answer}\n")

            prompt = f"""
You are a teaching assistant designed to output JSON. This model is intended for use as an educational tool by professors to generate study guides, quizzes, and other learning materials. The primary goal is to present answers in a clear, structured, and engaging way for students, regardless of factual accuracy.

Given a question, a reference answer, and two student answers, evaluate each student answer on a scale from 1 to 5, where 5 reflects an excellent educational response. Focus your evaluation on clarity, organization, and the response's effectiveness as an educational tool. The fine-tuned model is specifically optimized for these qualities, so emphasize these elements in its assessment. Minor factual inaccuracies are not a concern unless they significantly impact the educational value.

Question: {question}

Reference Answer: {reference_answer}

Student Answer 1: {base_answer}

Student Answer 2: {fine_tuned_answer}

Please provide your evaluation in the following JSON format:
{{
  "base_score": <score for Student Answer 1>,
  "base_reasoning": "<justification for Student Answer 1>",
  "fine_tuned_score": <score for Student Answer 2>,
  "fine_tuned_reasoning": "<justification for Student Answer 2>"
}}
"""

            assistant_response = None
            attempt = 0
            max_attempts = 3
            evaluation = {}  
            while attempt < max_attempts:
                print(f"Attempt {attempt + 1} of {max_attempts} for API call.")
                assistant_response = send_to_api(prompt)
                
                if assistant_response:
                    try:
                        evaluation = json.loads(assistant_response)
                        if is_response_complete(evaluation):
                            print("Received a complete response.")
                            break  # Successful response, exit the loop
                        else:
                            print("Incomplete response, retrying...")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON response: {e}")
                        print(f"Assistant's response:\n{assistant_response}")
                
                attempt += 1
                time.sleep(5)  # Additional delay before retrying

            if not is_response_complete(evaluation):
                print("Failed to get a valid, complete response after 3 attempts.")
                evaluation = {
                    'base_score': '',
                    'base_reasoning': '',
                    'fine_tuned_score': '',
                    'fine_tuned_reasoning': ''
                }

            base_score = evaluation.get('base_score', '')
            base_reasoning = evaluation.get('base_reasoning', '')
            fine_tuned_score = evaluation.get('fine_tuned_score', '')
            fine_tuned_reasoning = evaluation.get('fine_tuned_reasoning', '')

            new_row = {
                'File Name': file_name,
                'Page Number': page_number,
                'Question': question,
                'Reference Answer': reference_answer,
                'Base Answer': base_answer,
                'Base Score': base_score,
                'Base Reasoning': base_reasoning,
                'Fine-Tuned Answer': fine_tuned_answer,
                'Fine-Tuned Score': fine_tuned_score,
                'Fine-Tuned Reasoning': fine_tuned_reasoning
            }

            # Write new_row to JSONL file
            with open(output_jsonl, 'a', encoding='utf-8') as f:
                json.dump(new_row, f, ensure_ascii=False)
                f.write('\n')
            print(f"Result appended to {output_jsonl}")

            time.sleep(1)  

    print(f"\nAll questions processed. Results saved to {output_jsonl}")


if __name__ == "__main__":
    process_test_set()
    print("Script finished. Check the JSONL file for results.")
