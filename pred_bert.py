import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_PATH = "my_bert_model"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_map = {
    0: "SMS: Normal", 1: "SMS: SPAM",
    2: "IMDB: Neg", 3: "IMDB: Pos",
    4: "News: World", 5: "News: Sports",
    6: "News: Biz", 7: "News: Tech"
}

print("Loading BERT Model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print("✅ Model Loaded!")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=100)
    
    # Move inputs to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, pred_class = torch.max(probs, 1)
        
    return pred_class.item(), confidence.item() * 100

if __name__ == "__main__":
    print("Type 'exit' to quit.")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'exit': break
        
        idx, conf = predict(text)
        print(f"🤖 BERT says: {class_map[idx]} ({conf:.2f}%)")