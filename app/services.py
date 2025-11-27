import torch
import pickle
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

from model import get_model
from data_t import clean_text


LSTM_PATH = 'universal_model.pth'
VOCAB_PATH = 'vocab.pkl'
BERT_PATH = 'my_bert_model'

EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 8
MAX_SEQ_LENGTH = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_MAP = {
    0: "SMS: Normal", 1: "SMS: SPAM",
    2: "IMDB: Neg", 3: "IMDB: Pos",
    4: "News: World", 5: "News: Sports",
    6: "News: Biz", 7: "News: Tech"
}


lstm_model = None
lstm_vocab = None
bert_model = None
bert_tokenizer = None

def load_models():
    """
    Called once when server starts. Loads weights into global variables.
    """
    # Tell Python we want to write to the global variables
    global lstm_model, lstm_vocab, bert_model, bert_tokenizer
    
    print("---  SERVICES: Loading Artifacts... ---")

    # 1. LOAD LSTM
    try:
        with open(VOCAB_PATH, 'rb') as f:
            lstm_vocab = pickle.load(f)
            
        # Build empty brain
        lstm_model, _ = get_model(len(lstm_vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
        
        # Load weights
        try:
            state_dict = torch.load(LSTM_PATH, map_location=DEVICE, weights_only=True)
        except:
            state_dict = torch.load(LSTM_PATH, map_location=DEVICE)
            
        lstm_model.load_state_dict(state_dict)
        lstm_model.to(DEVICE)
        lstm_model.eval()
        print(" LSTM Loaded")
    except Exception as e:
        print(f" LSTM Load Failed: {e}")

    # 2. LOAD BERT
    try:
        bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH)
        bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_PATH)
        bert_model.to(DEVICE)
        bert_model.eval()
        print(" BERT Loaded")
    except Exception as e:
        print(f" BERT Load Failed: {e}")

def predict_lstm(text: str):
    if lstm_model is None: 
        return "LSTM Model not loaded"

    # 1. Tokenize
    tokens = clean_text(text)
    
    # 2. Map to IDs
    indices = [lstm_vocab.get(t, 1) for t in tokens]
    
    # 3. Pre-Padding (Zeros First)
    if len(indices) < MAX_SEQ_LENGTH:
        padded = ([0] * (MAX_SEQ_LENGTH - len(indices))) + indices
    else:
        padded = indices[-MAX_SEQ_LENGTH:]
        
    tensor = torch.tensor([padded], dtype=torch.long).to(DEVICE)
    
    # 4. Predict
    with torch.no_grad():
        output = lstm_model(tensor)
        probs = F.softmax(output, dim=1)
        conf, idx = torch.max(probs, 1)
        
    return {"label": CLASS_MAP[idx.item()], "conf": f"{conf.item()*100:.2f}%"}

def predict_bert(text: str):
    if bert_model is None: 
        return "Model not loaded"

    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=100)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        conf, idx = torch.max(probs, 1)
        
    return {"label": CLASS_MAP[idx.item()], "conf": f"{conf.item()*100:.2f}%"}

# In app/services.py (Add this to the bottom of the file)

def get_loaded_models():
    """
    Returns the loaded models and device from the global scope, 
    bypassing unreliable direct global imports.
    """
    # The global variables defined at the top of the file are returned here
    return lstm_model, bert_model, DEVICE


