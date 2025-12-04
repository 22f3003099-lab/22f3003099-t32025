import torch
import torch.nn as nn
from transformers import AutoModel

class BertForMultiLabel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", out_dim=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.pooler_output
        x = self.dropout(pooled)
        return self.fc(x)
