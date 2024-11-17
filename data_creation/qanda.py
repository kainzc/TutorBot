import os
import base64
import requests
import pdfplumber
import pandas as pd
from io import BytesIO
import json
import time

api_key = ""


pdf_dir = './test_data2'
output_csv = 'qa_pairs.csv'

def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")  
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def process_pdf(pdf_path):
    base64_images = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_image = page.to_image(resolution=300).original  
            base64_images.append((i + 1, encode_image(page_image)))
            print(f"Page {i+1} of {pdf_path} processed into base64.")
    return base64_images

def send_to_api(base64_images_batch, file_name):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
        {
            "role": "system",
            "content": "You are a helpful professor designed to output JSON. Given an image of document content, generate up to 5 question-answer pairs about the information presented. Produce 0-5 pairs based on the available information, ensuring questions are challenging without requiring direct references to the page or visible elements on it for solving. Return the results in JSON format as follows: {'qa_pairs': [{'question': question, 'answer': answer, 'subject': subject}, ...]}"

        },
        {
            "role": "user",
            "content": []
        }
    ]

    messages[1]['content'].append({
        "type": "text",
        "text": f"Generate question-answer pairs for the following pages of {file_name}. Include the subject for each pair."
    })

    for page_number, base64_image in base64_images_batch:
        messages[1]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 3500,
        "temperature": 0.7,
        "top_p": 0.95
    }

    print(f"Sending API request for {file_name}, pages {', '.join([str(p[0]) for p in base64_images_batch])}")
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        print("Response received:")
        print(response.json())
    else:
        print(f"Error response: {response.text}")

    return response.json()

def process_pdfs(pdf_directory, output_csv):
    # Check if the CSV already exists
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=['File Name', 'Page Number', 'Question', 'Answer', 'Subject'])

    api_call_count = 0  

    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"Processing file: {pdf_file}")
            base64_images = process_pdf(pdf_path)

            # Process 3 pages at a time
            for i in range(0, len(base64_images), 3):
                base64_images_batch = base64_images[i:i + 3]
                response = send_to_api(base64_images_batch, pdf_file)
                
                # Track API call and check for rate limiting
                api_call_count += 1
                if api_call_count % 3 == 0:
                    print("Pausing for 3 seconds to avoid rate limit...")
                    time.sleep(8)

                try:
                    if 'choices' in response:
                        content = response['choices'][0]['message']['content']
                        response_json = json.loads(content)

                        qa_pairs = response_json.get('qa_pairs', [])
                        print(f"Parsed Q&A pairs for {pdf_file}, pages {i + 1}-{i + len(base64_images_batch)}: {qa_pairs}")

                        for idx, qa_pair in enumerate(qa_pairs):
                            if idx < len(base64_images_batch):  
                                page_number = base64_images_batch[idx][0]
                                question = qa_pair.get('question', 'N/A')
                                answer = qa_pair.get('answer', 'N/A')
                                subject = qa_pair.get('subject', 'N/A')

                                temp_df = pd.DataFrame([{
                                    'File Name': pdf_file,
                                    'Page Number': page_number,
                                    'Question': question,
                                    'Answer': answer,
                                    'Subject': subject
                                }])

                                df = pd.concat([df, temp_df], ignore_index=True)
                    else:
                        print(f"Error: 'choices' not found in response for {pdf_file}, pages {i + 1}-{i + len(base64_images_batch)}")
                except Exception as e:
                    print(f"Error processing response for {pdf_file}, pages {i + 1}-{i + len(base64_images_batch)}: {e}")

    df.to_csv(output_csv, index=False)
    print(f"Q&A pairs saved to {output_csv}")


if __name__ == "__main__":
    process_pdfs(pdf_dir, output_csv)
    print(f"Script finished. Check {output_csv} for the results.")