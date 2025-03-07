import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super(MultiTaskModel, self).__init__()
        self.encoder = base_model 
        self.classifier = nn.Linear(384, 3) 
        self.sentiment = nn.Linear(384, 3)  

    def forward(self, embeddings):
        """
        embeddings: Input sentence embeddings from SentenceTransformer
        Returns: class_logits (category predictions), sentiment_logits (sentiment predictions)
        """
        class_logits = self.classifier(embeddings)
        sentiment_logits = self.sentiment(embeddings)
        return class_logits, sentiment_logits