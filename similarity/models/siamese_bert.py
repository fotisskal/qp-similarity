import torch
from transformers import AutoTokenizer, AutoModel

# Set up the Siamese architecture
class SiameseBERT(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert)
        self.bert = AutoModel.from_pretrained(bert)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, sentence1, sentence2):
        # Encode the two sentences using the Siamese-BERT model
        inputs1 = self.tokenizer(sentence1, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs2 = self.tokenizer(sentence2, padding=True, truncation=True, max_length=128, return_tensors="pt")
        outputs1 = self.bert(**inputs1)
        outputs2 = self.bert(**inputs2)
        pooled_output_1 = outputs1.last_hidden_state[:, 0, :]
        pooled_output_2 = outputs2.last_hidden_state[:, 0, :]

        # Compute the absolute difference between the pooled embeddings
        x = torch.abs(pooled_output_1 - pooled_output_2)

        # Apply a linear layer to obtain a similarity score
        similarity = torch.sigmoid(self.linear(x))
        return similarity