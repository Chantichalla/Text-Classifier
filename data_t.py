import torch
import re
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from collections import Counter
from data_loader import UniversalLoader
from sklearn.model_selection import train_test_split


MAX_VOCAB_LENGTH = 25000
MAX_SEQ_LENGTH = 100
BATCH_SIZE = 64

# For cleaning text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)# to only have letters and spaces and remove (',!, etc)
    return text.split()

def augment_data(texts, labels):
    """
    Takes long News/IMDB texts, chops them into short snippets,
    and adds them to the dataset. This cures 'Length Bias'.
    """
    print("   ✨ Augmenting Data (Creating short snippets)...")
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        # Add the original
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # If it is NOT SMS (Labels 0 and 1), create a short version
        # Labels 2,3 (IMDB) and 4,5,6,7 (News)
        if label >= 2:
            words = clean_text(text)
            
            # If the text is long enough, cut out the first 10-15 words
            if len(words) > 5:
                # Create a "Short Input" version
                short_snippet = " ".join(words[:12]) 
                augmented_texts.append(short_snippet)
                augmented_labels.append(label)
                
    return augmented_texts, augmented_labels
def build_vocab(text_list):
    print("building vocab")
    all_tokens =[]
    for text in (text_list):
        all_tokens.extend(clean_text(text))
    counts = Counter(all_tokens)
    vocab = {'<POD>':0,'<UNK>':1}
    sorted_list = sorted(counts, reverse=True, key=counts.get)[:MAX_VOCAB_LENGTH]

    for idx, word in enumerate(sorted_list):
        vocab[word] = idx+2
    return vocab


def text_to_tensor(text_list, vocab):
    numerical_list = []
    
    for i, text in enumerate(text_list):
        tokens = clean_text(text)
        indices = [vocab.get(t, 1) for t in tokens]
        
        current_len = len(indices)
        
        #  PADDING
        if current_len < MAX_SEQ_LENGTH:
            pad_count = MAX_SEQ_LENGTH - current_len
            padded = ([0] * pad_count) + indices 
        else:
            padded = indices[-MAX_SEQ_LENGTH:]


        if len(padded) != MAX_SEQ_LENGTH:
            print(f"❌ CRITICAL ERROR at index {i}")
            print(f"Original Length: {current_len}")
            print(f"Result Length: {len(padded)}")
            # This will tell us if the math broke
            print(f"Target Length was: {MAX_SEQ_LENGTH}")
            raise ValueError("Padding failed!")
            
        numerical_list.append(padded)
        
    return torch.tensor(numerical_list, dtype=torch.long)
def get_data_loader():
    loader = UniversalLoader(data_folder='data')
    #loading the data frame from previous class
    df = loader.combined_loader()
    # Two train test split, because we need three percentages like train,valuation and test .
    x_train_raw , x_test_temp , y_train_raw , y_test_temp = train_test_split(df['text'].tolist(),df['label'].tolist(),test_size=0.4, random_state=42)
    #Apply Data augmentation
    x_train , y_train = augment_data(x_train_raw, y_train_raw)
    x_test , x_val , y_test, y_val = train_test_split(x_test_temp,y_test_temp, test_size=0.5, random_state=42)
    # We only train x_train for this .
    vocab= build_vocab(x_train)
     # We convert the train ,val and test into torch with tenser converter (text_to_tensor)
    x_train_tensor = text_to_tensor(x_train, vocab)
    x_val_tensor = text_to_tensor(x_val, vocab)
    x_test_tensor = text_to_tensor(x_test, vocab)

    # now converting y into tensor.long because pytorch need to know that these are integers.
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    #TensorDataset behaves like a zipper. It zips the Questions (X) and Answers (y) together into pairs. When you ask for item 5, it gives you (Question 5, Answer 5)
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    val_data = TensorDataset(x_val_tensor, y_val_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    # Data loaders, we only shuffle the train data not the val and test data.
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, val_loader ,vocab

if __name__ ==  "__main__" :
    # Now we get 4 items 
    train_loader, val_loader , test_loader, vocab = get_data_loader()

    print("\nData Pipeline Ready!")
    print(f"Training Batches: {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}")
    print(f"Test Batches: {len(test_loader)}")