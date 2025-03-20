import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset

class BashAI(nn.Module):
    def __init__(self, hidden_size=256, output_size=None):  
        super(BashAI, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data(data, tokenizer):
    texts, labels = [], []
    unique_responses = sorted(set(str(item["response"]) for item in data))  # <-- sorted for consistency
    response_mapping = {response: idx for idx, response in enumerate(unique_responses)}

    for item in data:
        texts.append(str(item["input"]))
        labels.append(response_mapping[str(item["response"])])

    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels, dtype=torch.long)

    return encodings, labels, response_mapping

def train_model():
    with open("backend/ai_engine/bash_ai_dataset.json", "r") as f:
        data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset, labels, response_mapping = preprocess_data(data, tokenizer)

    tensor_dataset = TensorDataset(dataset["input_ids"], dataset["attention_mask"], labels)
    dataloader = DataLoader(tensor_dataset, batch_size=3, shuffle=True)

    model = BashAI(output_size=len(response_mapping))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        total_loss = 0
        for input_ids, attention_mask, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"🐺 Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "response_mapping": response_mapping
    }, "backend/ai_engine/bash_ai_model.pth")
    print("🐺 Training complete. Model saved.")

if __name__ == "__main__":
    train_model()
