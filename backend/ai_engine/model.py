import torch
import torch.nn as nn
from transformers import BertModel

# Load pre-trained BERT model once to avoid redownloading
bert = BertModel.from_pretrained("bert-base-uncased")

class BashAI(nn.Module):
    def __init__(self, hidden_size=256, output_size=None):
        super(BashAI, self).__init__()
        self.bert = bert
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits
