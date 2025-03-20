import torch
import torch.nn as nn
import torch.optim as optim
import random

# Import BashAI Model
from model import BashAI

# Dummy dataset (Replace with real conversations later)
training_data = [
    ("Hello", [1, 0, 0]),
    ("How are you?", [0, 1, 0]),
    ("Goodbye", [0, 0, 1])
]

# Convert text to numbers (basic example)
word_to_index = {word: i for i, (word, _) in enumerate(training_data)}

# Prepare data
X_train = torch.tensor([[word_to_index[word]] for word, _ in training_data], dtype=torch.float32)
y_train = torch.tensor([label for _, label in training_data], dtype=torch.float32)

# Initialize model
model = BashAI(input_size=1, hidden_size=8, output_size=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save trained model
torch.save(model.state_dict(), "bash_ai_model.pth")
print("Training complete. Model saved.")
