import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from data_t import get_data_loader, clean_text, text_to_tensor
from model import get_model

# --- CONFIGURATION ---
EMBED_DIM = 100
HIDDEN_DIM = 128 # Matches your training
OUTPUT_DIM = 8
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_diagnostic():
    print("--- 🏥 STARTING AI DIAGNOSTIC ---")
    
    # 1. Load the exact Validation Data used during training
    print("1. Loading Validation Data...")
    # We discard the new vocab this generates and load the saved one to be safe
    _, val_loader, _, _ = get_data_loader()
    
    # 2. Load the Saved Vocabulary
    print("2. Loading Saved Vocabulary...")
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    print(f"   Vocab Size: {len(vocab)}")

    # 3. Load the Model
    print("3. Loading Saved Model...")
    model, _ = get_model(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    try:
        model.load_state_dict(torch.load('universal_model.pth'))
    except:
        model.load_state_dict(torch.load('universal_model.pth', map_location='cpu'))
        
    model.to(device)
    model.eval()
    print("   ✅ Model Loaded.")

    # 4. Run Validation Loop (The Moment of Truth)
    print("\n4. Re-evaluating Model on Validation Data...")
    correct = 0
    total = 0
    
    # Class-wise accuracy tracking
    class_correct = list(0. for i in range(8))
    class_total = list(0. for i in range(8))
    
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\n" + "="*30)
    print(f"🏆 REAL ACCURACY: {100 * correct / total:.2f}%")
    print("="*30)
    
    print("\nClass-wise Breakdown:")
    classes = ["SMS Ham", "SMS Spam", "IMDB Neg", "IMDB Pos", "News World", "News Sport", "News Biz", "News Tech"]
    for i in range(8):
        if class_total[i] > 0:
            print(f"   {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")
            
    # 5. Test Manual Inputs (Sanity Check)
    print("\n5. Testing Manual Inputs (Short vs Long)...")
    
    long_review = "this movie was absolutely terrible i hated every minute of it the acting was garbage and the plot made no sense whatsoever i want my money back"
    short_review = "bad movie"
    
    inputs = [long_review, short_review]
    
    for text in inputs:
        # Manual Preprocessing matching predict.py
        tokens = clean_text(text)
        indices = [vocab.get(t, 1) for t in tokens]
        if len(indices) < MAX_SEQ_LENGTH:
            padded = indices + ([0] * (MAX_SEQ_LENGTH - len(indices)))
        else:
            padded = indices[:MAX_SEQ_LENGTH]
        
        tensor = torch.tensor([padded], dtype=torch.long).to(device)
        output = model(tensor)
        _, pred = torch.max(output, 1)
        print(f"   Input: '{text[:20]}...' -> Prediction: {classes[pred.item()]}")

if __name__ == "__main__":
    run_diagnostic()