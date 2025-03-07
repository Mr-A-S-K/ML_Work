import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from model.multi_task_model import MultiTaskModel
from data.sample_data import sentences, category_labels, sentiment_labels

# Load Pretrained SentenceTransformer model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Multi-task model
model = MultiTaskModel(base_model)

# Define Loss, Optimizer, Scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Create dataset and dataloader
sentence_embeddings = base_model.encode(sentences, convert_to_tensor=True)
dataset = TensorDataset(sentence_embeddings, category_labels, sentiment_labels)
train_loader = DataLoader(dataset, batch_size=3, shuffle=True)

# Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    for batch in train_loader:
        embeddings_batch, category_batch, sentiment_batch = batch
        optimizer.zero_grad()

        class_output, sentiment_output = model(embeddings_batch)

        class_loss = criterion(class_output, category_batch)
        sentiment_loss = criterion(sentiment_output, sentiment_batch)
        loss = class_loss + sentiment_loss

        loss.backward()
        optimizer.step()

    scheduler.step()

    # Print Progress Every 10 Epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Classification Loss: {class_loss.item():.4f}, Sentiment Loss: {sentiment_loss.item():.4f}")

print("Training Complete!")
