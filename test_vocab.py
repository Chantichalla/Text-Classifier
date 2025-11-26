import pickle
import os
from data_t import clean_text
VOCAB_PATH = 'vocab.pkl'

print(f"--- INSPECTING {VOCAB_PATH} ---")

if not os.path.exists(VOCAB_PATH):
    print("❌ ERROR: vocab.pkl does not exist!")
else:
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"✅ Vocab Loaded. Total words: {len(vocab)}")
    
    # Test specific words
    test_words = ["ronaldo", "movie", "free", "ticket", "government"]
    
    print("\n--- WORD LOOKUP TEST ---")
    for word in test_words:
        if word in vocab:
            print(f"✅ '{word}' is ID: {vocab[word]}")
        else:
            print(f"❌ '{word}' is NOT in the dictionary (Maps to 1)")

    print("\n--- TOKENIZATION TEST ---")
    sentence = "Ronaldo scored a goal"
    tokens = clean_text(sentence)
    indices = [vocab.get(t, 1) for t in tokens]
    print(f"Input: '{sentence}'")
    print(f"Tokens: {tokens}")
    print(f"IDs: {indices}")
    
    if all(id == 1 for id in indices):
        print("\n⚠️ MAJOR ISSUE: All words mapped to 1 (<UNK>). The model sees nothing!")
    else:
        print("\nINFO: Words are mapping to numbers. The Model weights might be the issue.")