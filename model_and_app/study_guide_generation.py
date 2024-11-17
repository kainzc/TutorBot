import os
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import json
import base64
import logging
from io import BytesIO
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, PaliGemmaForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AutoProcessor, AutoModelForImageTextToText

import gradio as gr
from huggingface_hub import login
import os
import json
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_path
import base64
import logging
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()

CACHE_DIR = "cache" 
os.makedirs(CACHE_DIR, exist_ok=True)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
login(token="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")


def load_images_from_pdfs(folder: str) -> List[Tuple[str, int, Image.Image]]:

    if not os.path.isabs(folder):
        raise ValueError(f"Provided folder path must be absolute. Got: {folder}")
    
    logger.debug(f"Loading PDFs from absolute folder path: {folder}")
    files = []
    
    for pdf_file in Path(folder).glob("*.pdf"):
        pdf_cache_path = os.path.join(CACHE_DIR, f"{pdf_file.stem}.json")
        if os.path.exists(pdf_cache_path):
            with open(pdf_cache_path, "r") as f:
                cached_data = json.load(f)
            for page_num, img_data in cached_data.items():
                image = Image.open(BytesIO(base64.b64decode(img_data)))
                files.append((pdf_file.stem, int(page_num), image))
        else:
            pages = convert_from_path(pdf_file, fmt="RGB")
            page_cache = {}
            for idx, page in enumerate(pages):
                buffered = BytesIO()
                page.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                page_cache[idx + 1] = img_str
                files.append((pdf_file.stem, idx + 1, page))
            with open(pdf_cache_path, "w") as f:
                json.dump(page_cache, f)
    
    return files




# Function for extracting content from images
def extract_content_from_images(pages: List[Tuple[str, int, Image.Image]]) -> List[Tuple[str, int, str]]:
    extracted_contents = []

    model_id = "google/paligemma-3b-ft-docvqa-896"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    for doc_name, page_num, image in pages:
        logger.debug(f"Processing document: {doc_name}, Page: {page_num}")

        prompt = ("<image><bos> Summarize the facts on this page.")
        

        image = image.convert("RGB").resize((896, 896))

        model_inputs = processor(text=prompt, images=image, return_tensors="pt")
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_inputs,  max_length=5000, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            extracted_contents.append((doc_name, page_num, decoded))



        

    return extracted_contents




import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the finetuned model path
FINETUNED_MODEL_PATH = r"C:\Users\ColeK\iCloudDrive\Computer Science\kaggleX\fine-tuning-experiments\model_and_app\finetuned_model"

# Load the tokenizer and model
finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
finetuned_model = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL_PATH,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16
).eval()

def generate_study_guide(query: str, extracted_contents: list, output_file: str) -> None:

    with open(output_file, "w", encoding="utf-8") as f:
        for doc_name, page_num, content in extracted_contents:
            # Prepare the prompt
            prompt = (
                f"Given the following content, create a relevant question-and-answer pair for the topic: '{query}'.\n\n"
                f"Content: {content}\n\n"
                "Provide a meaningful question and answer."
            )

            # Tokenize input
            inputs = finetuned_tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
            input_len = inputs["input_ids"].shape[-1]  # Determine input length

            # Generate output
            with torch.no_grad():
                outputs = finetuned_model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=500,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Extract only the generated portion
            generated_tokens = outputs[0][input_len:]
            qa_pair = finetuned_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Write to the output file
            f.write(f"Document: {doc_name}, Page: {page_num}\n{qa_pair}\n\n")

    print(f"Study guide saved to {output_file}")



def create_study_guide(query: str, output_file="study_guide.txt"):
    logger.debug(f"Creating study guide for query: {query}")
    files = load_images_from_pdfs("C:\\Users\\ColeK\\iCloudDrive\\Computer Science\\kaggleX\\fine-tuning-experiments\\model_and_app\\files")
    if not files:
        logger.error("No PDF files found in the 'files' directory.")
        return "No PDF files found in the 'files' directory."
    top_pages = files[:5]  # Adjust if you want to use relevance ranking
    extracted_contents = extract_content_from_images(top_pages)
    generate_study_guide(query, extracted_contents, output_file)
    return f"Study guide saved to {output_file}"

# Set up Gradio Interface
import gradio as gr

interface = gr.Interface(
    fn=lambda query: create_study_guide(query, output_file="study_guide.txt"),
    inputs=gr.Textbox(lines=2, placeholder="Enter the topic you want to study"),
    outputs="text",
    title="AI Study Guide Generator",
    description="Enter your study topic. The system will retrieve relevant pages and create a study guide for you."
)

if __name__ == "__main__":
    logger.debug("Starting Gradio interface...")
    interface.launch(debug=True)

