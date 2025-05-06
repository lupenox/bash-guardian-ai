from pathlib import Path

input_path = "/home/logan-lapierre/Desktop/Huge_Project/bash-guardian-ai/backend/ai_engine/gpt2_finetune/werewolf_subs_finetune_data.txt"  # Update this if needed
output_path = "bash_dataset_cleaned_final.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Remove duplicate lines that repeat too often
from collections import Counter

counts = Counter(lines)
filtered_lines = [line for line in lines if counts[line] < 5]  # tweak threshold if needed

# Group into conversation blocks
blocks = []
block = []
for line in filtered_lines:
    if line.startswith("<|user|>") or line.startswith("<|bash|>"):
        block.append(line)
        if len(block) >= 4 and block[-1].startswith("<|bash|>"):
            blocks.append("\n".join(block))
            block = []

# Save clean version
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(blocks))

print(f"✅ Cleaned dataset saved to: {output_path}")
