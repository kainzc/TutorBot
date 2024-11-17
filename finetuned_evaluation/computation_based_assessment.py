import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import nltk
nltk.download()

# Load and parse JSONL file
file_path = r'C:\Users\ColeK\iCloudDrive\Computer Science\kaggleX\fine-tuning-experiments\gemma_experiments\output.jsonl'
with open(file_path, 'r', encoding='utf-8') as file:  # Specify utf-8 encoding
    data = [json.loads(line.strip()) for line in file]


# Function to compute cosine similarity
def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Function to calculate keyword overlap
def compute_keyword_overlap(reference, answer):
    ref_tokens = set(word_tokenize(reference.lower()))
    ans_tokens = set(word_tokenize(answer.lower()))
    return len(ref_tokens & ans_tokens) / len(ref_tokens) if ref_tokens else 0

# Collect metrics for each entry
results = []
for entry in data:
    reference = entry.get("Reference Answer", "")
    base = entry.get("Base Answer", "")
    fine_tuned = entry.get("Fine-Tuned Answer", "")
    
    # Compute metrics
    base_similarity = compute_cosine_similarity(reference, base)
    fine_tuned_similarity = compute_cosine_similarity(reference, fine_tuned)
    base_keyword_overlap = compute_keyword_overlap(reference, base)
    fine_tuned_keyword_overlap = compute_keyword_overlap(reference, fine_tuned)
    
    # Record results
    results.append({
        "File Name": entry.get("File Name"),
        "Page Number": entry.get("Page Number"),
        "Base Similarity": base_similarity,
        "Fine-Tuned Similarity": fine_tuned_similarity,
        "Base Keyword Overlap": base_keyword_overlap,
        "Fine-Tuned Keyword Overlap": fine_tuned_keyword_overlap,
        "Base Length": len(base.split()),
        "Fine-Tuned Length": len(fine_tuned.split()),
        "Reference Length": len(reference.split())
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Print overall number breakdowns
print("Overall Averages and Standard Deviations:")
print(df_results.describe())

# Plot: Similarity Score Distribution
plt.figure(figsize=(10, 6))
plt.hist(df_results["Base Similarity"], bins=20, alpha=0.7, label="Base Similarity")
plt.hist(df_results["Fine-Tuned Similarity"], bins=20, alpha=0.7, label="Fine-Tuned Similarity")
plt.title("Similarity Score Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plot: Keyword Overlap Distribution
plt.figure(figsize=(10, 6))
plt.hist(df_results["Base Keyword Overlap"], bins=20, alpha=0.7, label="Base Keyword Overlap")
plt.hist(df_results["Fine-Tuned Keyword Overlap"], bins=20, alpha=0.7, label="Fine-Tuned Keyword Overlap")
plt.title("Keyword Overlap Distribution")
plt.xlabel("Overlap Ratio")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Bar Chart: Comparison of Overall Metrics
averages = {
    "Base Similarity": df_results["Base Similarity"].mean(),
    "Fine-Tuned Similarity": df_results["Fine-Tuned Similarity"].mean(),
    "Base Keyword Overlap": df_results["Base Keyword Overlap"].mean(),
    "Fine-Tuned Keyword Overlap": df_results["Fine-Tuned Keyword Overlap"].mean()
}

plt.figure(figsize=(10, 6))
plt.bar(averages.keys(), averages.values())
plt.title("Comparison of Overall Metrics")
plt.ylabel("Average Scores")
plt.xticks(rotation=45)
plt.show()
