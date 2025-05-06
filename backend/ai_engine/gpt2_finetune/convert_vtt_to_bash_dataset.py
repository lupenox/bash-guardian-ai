import os
import re
from pathlib import Path

# Folder with .vtt subtitle files
SUBS_DIR = Path.home() / "Desktop" / "werewolf_subs"
OUTPUT_FILE = Path.cwd() / "werewolf_subs_finetune_data.txt"

def clean_line(line):
    # Remove timestamps, tags, and junk
    line = re.sub(r"<[^>]+>", "", line)  # Remove HTML-like tags
    line = line.strip()
    return line

def parse_vtt(file_path):
    dialogue = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        current_block = []
        for line in lines:
            line = line.strip()
            if "-->" in line or not line:
                continue
            if re.match(r"^\d+$", line):  # skip numeric cue indexes
                continue
            cleaned = clean_line(line)
            if cleaned:
                dialogue.append(cleaned)
    return dialogue

def pair_lines(lines):
    pairs = []
    for i in range(len(lines) - 1):
        prompt = lines[i]
        response = lines[i + 1]
        if 10 < len(prompt) < 200 and 10 < len(response) < 300:
            pair = f"<|user|> {prompt}\n<|bash|> {response}"
            pairs.append(pair)
    return pairs

def main():
    all_pairs = []
    for file in SUBS_DIR.glob("*.vtt"):
        print(f"🐾 Processing: {file.name}")
        lines = parse_vtt(file)
        pairs = pair_lines(lines)
        all_pairs.extend(pairs)

    print(f"✨ Extracted {len(all_pairs)} prompt-response pairs.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for pair in all_pairs:
            out.write(pair + "\n")

    print(f"✅ Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
