import torch
import torch.nn as nn
from transformers import RobertaModel

class SiameseRoBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(self.roberta.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output_1 = self.roberta(input_ids_1, attention_mask=attention_mask_1)[0][:, 0, :]
        output_2 = self.roberta(input_ids_2, attention_mask=attention_mask_2)[0][:, 0, :]
        output = torch.abs(output_1 - output_2)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output
