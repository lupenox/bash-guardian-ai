import torch
import torch.nn as nn
import torch.optim as optim

class BashAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BashAI, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

# Example usage
if __name__ == "__main__":
    model = BashAI(input_size=100, hidden_size=256, output_size=10)
    print(model)
