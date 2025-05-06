import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model and tokenizer from fine-tuned output
MODEL_DIR = "backend/ai_engine/gpt2_finetune/output/checkpoint-282"

print("🐺 Loading Bash's trained werewolf soul...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()

if torch.cuda.is_available():
    model.to("cuda")
    print("🚀 Using CUDA for generation")

print("✨ Talk to Bash! Type 'exit' to leave.")

while True:
    user_input = input("\n<|user|> ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("🐾 Bash: I'll be right here when you need me, okay?")
        break

    prompt = f"<|user|> {user_input}\n<|bash|>"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.85,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract just Bash's reply
    if "<|bash|>" in response:
        bash_reply = response.split("<|bash|>")[-1].strip()
        if "<|user|>" in bash_reply:
            bash_reply = bash_reply.split("<|user|>")[0].strip()
        print(f"🐾 Bash: {bash_reply}")
    else:
        print("🐾 Bash: ...I'm thinking, pup.")
