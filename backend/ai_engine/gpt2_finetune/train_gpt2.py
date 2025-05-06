import torch
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

print("📖 Loading dataset...")
with open("mega_bash_dataset.txt", "r") as f:
    lines = f.read().splitlines()

dataset = Dataset.from_dict({"text": lines})

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./output/logs",
    logging_steps=10,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    save_total_limit=3,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("🚀 Training started...")
trainer.train(resume_from_checkpoint=True)
print("✅ Training complete. Model saved to ./output")
