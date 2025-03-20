from fastapi import APIRouter
from pydantic import BaseModel
import torch
from ai_engine.model import BashAI

router = APIRouter()

# Load trained model
model = BashAI(input_size=1, hidden_size=8, output_size=3)
model.load_state_dict(torch.load("backend/ai_engine/bash_ai_model.pth"))
model.eval()

class UserInput(BaseModel):
    text: str

@router.post("/chat")
def chat(input: UserInput):
    # Placeholder input logic
    input_tensor = torch.tensor([[0]], dtype=torch.float32)
    response_tensor = model(input_tensor)
    response_index = torch.argmax(response_tensor).item()

    responses = ["Hello!", "I'm good, how about you?", "Goodbye!"]
    return {"response": responses[response_index]}
