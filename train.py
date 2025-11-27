import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle
import numpy as np
import os

from data_t import get_data_loader
from model import LSTM

#Configuration
NUM_EPOCHS = 20 # How many times to read the  entire dataset
LEARNING_RATE = 0.0001 # How fast the model learns
HIDDEN_DIM = 128
N_LAYERS = 2
EMBED_DIM = 100
OUTPUT_DIM = 8
#Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# In train.py

# --- NEW HELPER: GLOVE LOADER ---
def load_glove_vectors(vocab, embed_dim):
    # GloVe file name for 100 dimensions
    GLOVE_FILE = 'glove.6B.100d.txt'
    
    # Check if the file exists first
    if not os.path.exists(GLOVE_FILE):
        print(f"\n❌ GLOVE ERROR: '{GLOVE_FILE}' not found. Download it and place it in the root directory.")
        return None, None

    print(f"Loading GloVe vectors from {GLOVE_FILE}...")
    
    # 1. Initialize a matrix with random numbers (this will hold the GloVe vectors)
    # Shape: [Vocab Size, Embedding Dimension]
    glove_weights = np.random.uniform(-0.05, 0.05, (len(vocab), embed_dim))
    
    # 2. Create a dictionary to map words to their GloVe vectors
    glove_map = {}
    with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove_map[word] = vector
            
    # 3. Map GloVe vectors to our Vocab IDs
    hits = 0
    
    # Iterate through our custom vocabulary {word: ID}
    for word, idx in vocab.items():
        if word in glove_map:
            # We found the word! Inject the GloVe vector into our matrix
            glove_weights[idx] = glove_map[word]
            hits += 1
        elif word in ['<PAD>', '<UNK>']:
            # Ensure padding/unknown tokens are zeroed out (or left as random init)
            glove_weights[idx] = np.zeros(embed_dim, dtype=np.float32)
            hits += 1
            
    print(f"   ✅ Successfully mapped {hits} out of {len(vocab)} words.")
    
    # Convert NumPy array to PyTorch Tensor
    glove_tensor = torch.tensor(glove_weights, dtype=torch.float)
    
    return glove_tensor

def train_model():
    # GET Data
    train_loader, val_loader , test_loader , vocab = get_data_loader()
    vocab_size = len(vocab)
    print(f"Vocab size:{vocab_size}")
    # 2. LOAD GLOVE WEIGHTS 
    glove_tensor = load_glove_vectors(vocab, EMBED_DIM)
    if glove_tensor is None:
        glove_tensor = torch.empty(vocab_size, EMBED_DIM) # Placeholder

    #Build model
    model = LSTM(vocab_size , EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, glove_tensor)
    #Moving brain to GPU
    model = model.to(device)
    #Setup tools
    criterion = nn.CrossEntropyLoss()

    #Optimizer : the teacher that optimizes the weights(Adam is the best one)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #The training Loop
    print("Training Started")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # A. Training Phase
        model.train()
        total_train_loss=0
        correct_predictions=0
        total_samples = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            #Zero gradients (Clears previous correction)
            optimizer.zero_grad()

            #The guess
            predictions = model(texts)

            #Calculate Loss
            loss = criterion(predictions, labels)

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

            #Get the index of highest score
            _, predicted_class = torch.max(predictions, 1)
            correct_predictions += (predicted_class == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate training Accuracy
        train_acc = (correct_predictions/ total_samples) * 100
        avg_train_loss = total_train_loss / len(train_loader)

        # B. Validation Phase
        model.eval() # turn off learning mode
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for texts, labels in val_loader :
                texts , labels = texts.to(device), labels.to(device)

                preds = model(texts)

                loss = criterion(preds, labels)

                total_val_loss += loss.item()
                _, predicted_class = torch.max(preds, 1)
                val_correct += (predicted_class == labels).sum().item()
                val_total += labels.size(0)
        val_acc = (val_correct / val_total) * 100
        avg_val_loss= total_val_loss/ len(val_loader)

        # C.Print Results
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02}| Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\t Train Loss: {avg_train_loss:.3f} | Train Acc : {train_acc:.2f}%")
        print(f"\tVal Loss : {avg_val_loss:.3f} | Val. Acc: {val_acc:.2f}%")

    print("\tTraining Complete!, saving model")
    torch.save(model.state_dict(), 'universal_model.pth')
    print("MOdel Saved as 'universal_model.pth'")
    # <--- ADD THIS BLOCK TO SAVE VOCAB ---
    print("Saving Vocabulary...")
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print(" Vocab saved as 'vocab.pkl'")


if __name__ == "__main__":
    train_model()