import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer
from ai_engine.model import BashAI

app = FastAPI()

# Only load tokenizer and trained model once
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
checkpoint = torch.load("backend/ai_engine/bash_ai_model.pth", map_location=torch.device("cpu"))

response_mapping = checkpoint["response_mapping"]
response_list = list(response_mapping.keys())

# Load model from checkpoint (no training here!)
model = BashAI(output_size=len(response_mapping))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class UserInput(BaseModel):
    text: str

@app.post("/chat")
def chat(input: UserInput):
    encoding = TOKENIZER(input.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        response_tensor = model(encoding["input_ids"], encoding["attention_mask"])
        response_index = torch.argmax(response_tensor).item()
    return {"response": response_list[response_index]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
