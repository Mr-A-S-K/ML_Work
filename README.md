# **Multi-Task Learning with Sentence Transformers**  

## **Overview**  
This project implements a **multi-task learning** model using **Sentence Transformers** to classify sentences into categories and predict sentiment. We use **PyTorch** for model training and **Docker** for containerization.  

## **Tasks & Design Choices**  

### **1. Sentence Transformer Implementation**  
- We use **all-MiniLM-L6-v2**, a lightweight yet powerful sentence embedding model from **Sentence Transformers**.  
- This model generates 384-dimensional embeddings that serve as inputs to our classifier and sentiment predictor.  
- **Why?**  
  - It balances efficiency and accuracy.  
  - It provides semantically meaningful embeddings without requiring extensive fine-tuning.  

### **2. Multi-Task Learning Expansion**  
- Our model has a **shared transformer encoder** and two separate **fully connected (FC) layers** for:  
  - **Category classification** (Tech, Sports, Politics).  
  - **Sentiment classification** (Positive, Negative, Neutral).  
- **Why?**  
  - Multi-task learning improves generalization.  
  - It allows shared representations, reducing overfitting.  

### **3. Training Considerations**  
- We explore different training strategies:  
  - **Freezing the entire network:** Efficient but prevents fine-tuning.  
  - **Freezing only the transformer backbone:** Allows fine-tuning of classification layers, balancing efficiency and adaptation.  
  - **Freezing only one task head:** Useful if one task is more critical and the other is overfitting.  
- **Why?**  
  - We enable **fine-tuning** to adapt to our specific dataset.  
  - This improves classification accuracy while leveraging transfer learning.  

### **4. Training Loop Implementation**  
- **Loss Function:** CrossEntropyLoss for both tasks.  
- **Optimizer:** Adam (learning rate = 0.0001).  
- **Learning Rate Scheduler:** StepLR to reduce learning rate over time.  
- **Mini-batch Training:** Uses DataLoader for efficient training.  
- **Why?**  
  - Stochastic gradient descent improves convergence.  
  - Learning rate decay prevents overfitting.  

---

## **Running the Code**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/multi-task-learning.git
cd multi-task-learning
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Run the Training Script**  
```bash
python train.py
```

### **4. Test the Model**  
```bash
python test.py
```

### **5. (Optional) Run in Docker**  
```bash
docker build -t multitask_model .
docker run multitask_model
```

---

## **Training Considerations (Detailed Explanation)**  

### **1. Freezing the Entire Network**  
- The transformer remains unchanged, and only the final layers are trained.  
- **Advantages:**  
  - Faster training.  
  - Requires less labeled data.  
- **Disadvantages:**  
  - The model may not adapt well to our dataset.  

### **2. Freezing Only the Transformer Backbone**  
- The transformer model is frozen, but the task-specific heads are trainable.  
- **Advantages:**  
  - Maintains pretrained knowledge.  
  - Allows slight adaptation to our dataset.  
- **Disadvantages:**  
  - May not capture domain-specific nuances effectively.  

### **3. Freezing Only One Task-Specific Head**  
- One task (e.g., sentiment classification) is frozen while the other (e.g., category classification) is trained.  
- **Advantages:**  
  - Helps balance learning when one task is more challenging.  
  - Prevents catastrophic forgetting.  
- **Disadvantages:**  
  - Can limit generalization for the frozen task.  

### **Transfer Learning Approach**  
- **Pretrained Model Choice:** `all-MiniLM-L6-v2` (fast, efficient, and accurate).  
- **Layers to Freeze/Unfreeze:**  
  - Freeze the transformer backbone for early epochs.  
  - Fine-tune the entire model after convergence.  
- **Rationale:**  
  - Initially, the frozen model retains general knowledge.  
  - Gradually unfreezing allows adaptation to our dataset without overfitting.  

---

## **Conclusion**  
This project demonstrates how multi-task learning and transfer learning can enhance text classification. The fine-tuning strategy balances efficiency and accuracy, making it practical for real-world applications. 🚀  
