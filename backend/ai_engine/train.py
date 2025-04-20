import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from ai_engine.model import BashAI

# Load dataset
with open("backend/ai_engine/bash_ai_dataset.json", "r") as f:
    data = json.load(f)

# Expanded dataset
expanded_data = [
    {"input": "I'm feeling overwhelmed.", "response": "Easy, pup. Deep breaths. I'm right here with you. Focus on my voice.", "tone": "comforting"},
    {"input": "Do you love me?", "response": "You're my pack. My world. Of course I do.", "tone": "reassuring"},
    # (rest of expanded_data goes here)
]

data.extend(expanded_data)

# Preprocess
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(data):
    texts, labels = [], []

    unique_responses = set(str(item["response"]) for item in data)
    response_mapping = {str(response): idx for idx, response in enumerate(unique_responses)}

    for idx, item in enumerate(data):
        input_text = str(item["input"])
        response_text = str(item["response"])
        mapped_label = response_mapping[response_text]

        texts.append(input_text)
        labels.append(mapped_label)

    encodings = TOKENIZER(texts, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels, dtype=torch.long)

    return encodings, labels, response_mapping

dataset, labels, response_mapping = preprocess_data(data)
tensor_dataset = TensorDataset(dataset["input_ids"], dataset["attention_mask"], labels)
dataloader = DataLoader(tensor_dataset, batch_size=3, shuffle=True)

# Train
model = BashAI(output_size=len(response_mapping))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, batch_labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"🐺 Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")

# Save
torch.save({
    "model_state_dict": model.state_dict(),
    "response_mapping": response_mapping
}, "backend/ai_engine/bash_ai_model.pth")

print("🐺 Training complete. Bash AI is ready to chat!")
