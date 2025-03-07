import torch

# Sample sentences
sentences = [
    "I love Datascience",
    "Quantum computing is fun.",
    "AI might take over the jobs",
    "The team played amazing",
    "He scored a goal",
    "The player broke his arm",
    "The government passed a new law",
    "The election results are out",
    "The election results are disappointing"
]

# Labels for categories (Tech, Sports, Politics)
category_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)

# Labels for sentiment (Positive, Negative, Neutral)
sentiment_labels = torch.tensor([2, 2, 0, 2, 2, 0, 1, 1, 0], dtype=torch.long)
