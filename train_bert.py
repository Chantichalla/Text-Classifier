import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from data_loader import UniversalLoader # We reuse your original loader!
import time

# --- CONFIGURATION ---
# DistilBERT is smaller and faster than full BERT
MODEL_NAME = 'distilbert-base-uncased'
BATCH_SIZE = 16 # BERT is heavy, so we use smaller batches to save GPU RAM
EPOCHS = 3      # BERT learns FAST. 3 epochs is usually enough.
LEARNING_RATE = 2e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training on: {device}")

# Data Augmentation
def augment_data(texts, labels):
    """
    Takes long News/IMDB texts, chops them into short snippets,
    and adds them to the dataset.
    """
    print("   ✨ Augmenting Data (Creating short snippets)...")
    augmented_texts = []
    augmented_labels = []
    
    # Simple cleaner to split by word count
    def get_words(t): return str(t).split()
    
    for text, label in zip(texts, labels):
        # Always keep the original
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # If it is NOT SMS (Labels 0 and 1)
        # We want to create "Fake Short News" and "Fake Short Reviews"
        if label >= 2:
            words = get_words(text)
            
            # If the text is long enough, take the first 10 words
            if len(words) > 8:
                short_snippet = " ".join(words[:10]) 
                augmented_texts.append(short_snippet)
                augmented_labels.append(label)

    return augmented_texts, augmented_labels
# --- 1. PREPARE DATASET CLASS ---
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # The Tokenizer handles [CLS], [SEP], Padding, and Attention Masks automatically!
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    # --- 2. LOAD DATA ---
    print("Loading Data...")
    loader = UniversalLoader(data_folder='data')
    df = loader.combined_loader()
    
    # Split Data
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )
    # --- APPLY AUGMENTATION ---
    print(f"Original Training Size: {len(X_train_raw)}")
    X_train, y_train = augment_data(X_train_raw, y_train_raw)
    print(f"Augmented Training Size: {len(X_train)} (Added short samples)")
    # --- 3. SETUP BERT ---
    print("Downloading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=8)
    model = model.to(device)

    # Create Datasets
    train_dataset = BertDataset(X_train, y_train, tokenizer)
    val_dataset = BertDataset(X_val, y_val, tokenizer)

    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 4. TRAINING LOOP ---
    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward Pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward Pass
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        val_acc = val_correct / val_total * 100
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {(time.time()-start_time)/60:.1f} min")

    # --- 5. SAVE ---
    print("\nSaving BERT Model...")
    model.save_pretrained("my_bert_model")
    tokenizer.save_pretrained("my_bert_model")
    print("✅ Saved to folder 'my_bert_model/'")

if __name__ == "__main__":
    train()