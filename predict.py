import torch
import pickle
import sys
import os
from data_t import clean_text
from model import get_model

# --- CHECK CONFIGURATION ---
# DID YOU CHANGE HIDDEN_DIM IN TRAIN.PY? 
# IF YES, YOU MUST CHANGE IT HERE TOO!
EMBED_DIM = 100
HIDDEN_DIM = 128   # <--- Checking if this matches your new training!
OUTPUT_DIM = 8
MAX_SEQ_LENGTH = 100
MODEL_PATH = 'universal_model.pth'
VOCAB_PATH = 'vocab.pkl'

class_map = {
    0: "SMS: Normal", 1: "SMS: SPAM",
    2: "IMDB: Negative", 3: "IMDB: Positive",
    4: "News: World", 5: "News: Sports",
    6: "News: Business", 7: "News: Sci/Tech"
}

def load_artifacts():
    print("--- LOADING ARTIFACTS ---")
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
        print("❌ Missing files. Run train.py first.")
        sys.exit()

    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print(f"✅ Dictionary Size: {len(vocab)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = get_model(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    except:
        model.load_state_dict(torch.load(MODEL_PATH))
        
    model.to(device)
    model.eval()
    return model, vocab, device

def predict_with_debug(text, model, vocab, device):
    # 1. Tokenize
    tokens = clean_text(text)
    
    # 2. Convert to IDs (DEBUGGING STEP)
    indices = [vocab.get(t, 1) for t in tokens]
    
    print(f"\n🔍 DEBUG INFO:")
    print(f"   Input: '{text}'")
    print(f"   Tokens: {tokens}")
    print(f"   IDs: {indices}") # <--- THIS IS WHAT WE NEED TO SEE
    
    # Check how many are UNK (ID 1)
    unk_count = indices.count(1)
    if unk_count > 0:
        print(f"   ⚠️ WARNING: {unk_count} words are Unknown (ID 1)!")

    # 3. Pad
    if len(indices) < MAX_SEQ_LENGTH:
        padded = ([0] * (MAX_SEQ_LENGTH - len(indices))) + indices
    else:
        padded = indices[-MAX_SEQ_LENGTH:]

    # 4. Predict
    tensor_input = torch.tensor([padded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        prediction = model(tensor_input)
        probs = torch.nn.functional.softmax(prediction, dim=1)
        confidence, top_class = torch.max(probs, 1)
        
        return top_class.item(), confidence.item() * 100

if __name__ == "__main__":
    model, vocab, device = load_artifacts()
    print("\nType 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        idx, conf = predict_with_debug(user_input, model, vocab, device)
        print(f"🤖 Prediction: {class_map[idx]} ({conf:.2f}%)")