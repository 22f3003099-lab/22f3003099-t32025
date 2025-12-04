import torch
import torch.nn as nn
from transformers import AutoModel

def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return (hidden_states * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)

def max_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    hidden_states[mask == 0] = -1e9
    return hidden_states.max(1).values

class EmotionClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", output_dim=5, pooling='cls'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = outputs.pooler_output
        elif self.pooling == "mean":
            pooled = mean_pooling(hidden, attention_mask)
        elif self.pooling == "max":
            pooled = max_pooling(hidden, attention_mask)
        else:
            raise ValueError("Invalid pooling type")

        return self.fc(self.dropout(pooled))
