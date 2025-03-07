import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from model.multi_task_model import MultiTaskModel

# Load Pretrained SentenceTransformer model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Multi-task model
model = MultiTaskModel(base_model)
model.load_state_dict(torch.load("model.pth"))  # Assuming the model was saved

# Test the model
test_sentences = [
    "I am excited about this new technology!",
    "The game was disappointing.",
    "The government passed a new law."
]

test_embeddings = base_model.encode(test_sentences, convert_to_tensor=True)

class_output, sentiment_output = model(test_embeddings)

class_probs = F.softmax(class_output, dim=1)
sentiment_probs = F.softmax(sentiment_output, dim=1)

class_preds = torch.argmax(class_probs, dim=1)
sentiment_preds = torch.argmax(sentiment_probs, dim=1)

class_labels = ["Tech", "Sports", "Politics"]
sentiment_labels = ["Negative", "Neutral", "Positive"]

for i, sentence in enumerate(test_sentences):
    print(f"Sentence: {sentence}")
    print(f"Predicted Category: {class_labels[class_preds[i]]}")
    print(f"Predicted Sentiment: {sentiment_labels[sentiment_preds[i]]}")
    print("-" * 50)
