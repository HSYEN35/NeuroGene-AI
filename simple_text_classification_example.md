## Simple Text Data Collection and Labeling Architecture Example with PyTorch

This small example demonstrates how an AI system can collect text-based data from its environment, perform basic preprocessing and labeling to create its own dataset, and use PyTorch to train a simple classification model as an introductory demo and architectural design.

### Steps included:

- Simulated text data  
- Preprocessing (cleaning, tokenization, padding)  
- Building a simple vocabulary  
- Preparing a labeled dataset  
- Defining an embedding and fully connected classifier in PyTorch  
- Training the model and making predictions  

This example can be extended with much more complex data collection, preprocessing, and labeling systems in real-world projects.

---

import torch
import torch.nn as nn
import torch.optim as optim
import re

# --- 1. Veri Toplama (Simülasyon) ---
raw_texts = [
    "Hava bugün çok güzel, dışarı çıkalım!",
    "Yağmur yağıyor, evde kalmalıyız.",
    "Projeyi yarın teslim etmeliyiz.",
    "Bugün biraz stresliyim, çok işim var.",
    "Tatilde çok dinlendim, enerji doluyum!"
]

# --- 2. Ön İşleme ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zığüşöçİĞÜŞÖÇ\s]', '', text)  # Türkçe karakterleri koruyarak temizle
    return text.strip()

texts = [preprocess(t) for t in raw_texts]

# --- 3. Basit Kelime Sözlüğü ve Tokenizasyon ---
vocab = {}
for sentence in texts:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)+1  # 0 rezerv, 1'den başla

def tokenize(text):
    return [vocab.get(word, 0) for word in text.split()]

tokenized_texts = [tokenize(t) for t in texts]

# --- 4. Basit Padding Fonksiyonu ---
def pad_sequence(seq, max_len=10):
    return seq + [0]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

max_len = 10
padded_texts = [pad_sequence(t, max_len) for t in tokenized_texts]

# --- 5. Etiketler (Simülasyon) ---
# Örnek olarak 0: Olumlu duygu, 1: Olumsuz duygu
labels = torch.tensor([0,1,1,1,0])  # Basit örnek etiketler

# --- 6. Model Tanımı ---
class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim*max_len, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)           # (batch_size, max_len, embed_dim)
        x = x.view(x.size(0), -1)      # Flatten
        x = self.fc(x)
        return x

vocab_size = len(vocab)
embed_dim = 8
num_classes = 2

model = SimpleTextClassifier(vocab_size, embed_dim, num_classes)

# --- 7. Eğitim Ayarları ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- 8. Eğitim Döngüsü ---
inputs = torch.tensor(padded_texts)
targets = labels

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

# --- 9. Test Tahmini ---
test_sentence = "Bugün çok mutluyum"
test_processed = preprocess(test_sentence)
test_tokenized = tokenize(test_processed)
test_padded = pad_sequence(test_tokenized, max_len)
test_input = torch.tensor([test_padded])

model.eval()
with torch.no_grad():
    pred = model(test_input)
    predicted_label = torch.argmax(pred, dim=1).item()

print(f"\"{test_sentence}\" cümlesinin tahmini etiketi: {predicted_label} (0=Olumlu,1=Olumsuz)")
