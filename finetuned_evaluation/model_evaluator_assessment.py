import json
import matplotlib.pyplot as plt

# Specify the path to your JSONL file
filename = r''

# Lists to store the scores
base_scores = []
fine_tuned_scores = []

# Read the JSONL file and extract the scores
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        base_score = data.get('Base Score')
        fine_tuned_score = data.get('Fine-Tuned Score')
        if base_score is not None:
            base_scores.append(base_score)
        if fine_tuned_score is not None:
            fine_tuned_scores.append(fine_tuned_score)

# Check if scores were extracted
if not base_scores or not fine_tuned_scores:
    print("No scores found in the data.")
    exit()

# Calculate average scores
base_mean = sum(base_scores) / len(base_scores)
fine_tuned_mean = sum(fine_tuned_scores) / len(fine_tuned_scores)

# Display average scores
print(f"Base Model Average Score: {base_mean:.2f}")
print(f"Fine-Tuned Model Average Score: {fine_tuned_mean:.2f}")

# Count how many times the fine-tuned model outperformed the base model
improved_count = sum(1 for b, f in zip(base_scores, fine_tuned_scores) if f > b)
equal_count = sum(1 for b, f in zip(base_scores, fine_tuned_scores) if f == b)
decreased_count = sum(1 for b, f in zip(base_scores, fine_tuned_scores) if f < b)
total = len(base_scores)

# Display improvement statistics
print(f"\nTotal Entries: {total}")
print(f"Fine-Tuned Model Improved: {improved_count} times ({(improved_count / total) * 100:.2f}%)")
print(f"Scores Remained Equal: {equal_count} times ({(equal_count / total) * 100:.2f}%)")
print(f"Fine-Tuned Model Decreased: {decreased_count} times ({(decreased_count / total) * 100:.2f}%)")

# Create a bar chart for score distribution comparison
categories = ['Improved', 'Equal', 'Decreased']
counts = [improved_count, equal_count, decreased_count]

plt.figure(figsize=(10, 6))
plt.bar(categories, counts, color=['green', 'blue', 'red'])

# Adding annotations for clarity
for i, count in enumerate(counts):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10)

# Titles and labels
plt.title('Comparison of Fine-Tuned Model Performance', fontsize=14)
plt.xlabel('Performance Category', fontsize=12)
plt.ylabel('Number of Entries', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the chart
plt.show()
