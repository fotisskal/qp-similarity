import torch
from transformers import AutoModel, AutoTokenizer

class SentenceBERT(torch.nn.Module):
    def __init__(self, model_name):
        super(SentenceBERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        output = self.model(**encoded_input)
        pooled_output = output[1]
        return pooled_output