print("🔥 Training script started")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Paths and names
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_path = "dataset/bash_asmr_dataset.jsonl"
output_dir = "../bash-lora"

print("📦 Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path, split="train")
print(f"✅ Dataset loaded with {len(dataset)} samples")

print("🧠 Loading tokenizer and model (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Tokenizer loaded")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
print("✅ Model loaded")

# Preprocess dataset
print("🔧 Tokenizing dataset...")
def format(example):
    prompt = f"<|system|>\n{example['system']}\n<|user|>\n{example['input']}\n<|assistant|>\n{example['output']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(format)
print("✅ Tokenization complete")

# LoRA config
print("🛠️ Applying LoRA configuration...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
print("✅ LoRA adapter applied")

# Training config
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    optim="adamw_torch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("🚀 Starting training...")
trainer.train()
print("🎉 Training complete!")

# Save adapter
print("💾 Saving trained adapter...")
model.save_pretrained(output_dir)
print(f"✅ Adapter saved to {output_dir}")
