"""Encoder condiviso + due teste: BIO per i dati personali, multi-label per i contenuti dannosi."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SensitiveClassifier(nn.Module):
    def __init__(self, encoder_name, n_bio, n_safety, dropout=0.1, attn="sdpa"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, attn_implementation=attn, add_pooling_layer=False)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pii_head = nn.Linear(h, n_bio)
        self.safety_head = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Dropout(dropout), nn.Linear(h, n_safety))

    def forward(self, input_ids, attention_mask, task):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if task == "pii":
            return self.pii_head(self.dropout(hs))
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)  # mean pooling
        return self.safety_head(self.dropout(pooled))

    @staticmethod
    def loss(logits, labels, task):
        if task == "pii":
            return F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return F.binary_cross_entropy_with_logits(logits.float(), labels)
