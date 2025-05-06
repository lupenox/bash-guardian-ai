from pathlib import Path

input_path = "/home/logan-lapierre/Desktop/Huge_Project/bash-guardian-ai/backend/ai_engine/gpt2_finetune/werewolf_subs_finetune_data.txt"
output_dir = Path("bash_chunks")
output_dir.mkdir(exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

blocks = content.split("\n\n")  # Each conversation block is separated by two newlines
chunk_size = 100  # number of conversation blocks per chunk

for i in range(0, len(blocks), chunk_size):
    chunk = blocks[i:i+chunk_size]
    filename = output_dir / f"bash_chunk_{i//chunk_size + 1}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunk))

print(f"✅ Split into {len(list(output_dir.glob('*.txt')))} chunks in folder: {output_dir}")
