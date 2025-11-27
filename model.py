import torch.nn as nn
import torch

# Global parameters
N_LAYERS = 2
OUTPUT_DIM = 8
HIDDEN_DIM = 128
DROPOUT = 0.5

class LSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim, hidden_dim, output_dim, glove_weights=None):
        super(LSTM,self).__init__()
        #First layer : Embed layer
        if glove_weights is not None :
            self.embedding = nn.Embedding.from_pretrained(glove_weights, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        #Second layer : LSTM layer
        self.lstm = nn.LSTM(embed_dim,HIDDEN_DIM,num_layers=N_LAYERS, batch_first=True, bidirectional=True, dropout= DROPOUT)

        #Third layer: FUlly connected output
        self.fc = nn.Linear(HIDDEN_DIM * 2, OUTPUT_DIM)
        
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self,text):
        embedded = self.dropout(self.embedding(text))

        lstm_output, (hidden, cell) = self.lstm(embedded)

        backward_hidden = hidden[-2, :, :]
        forward_hidden = hidden[-1, :, :]

        final_summary = torch.cat((backward_hidden,forward_hidden), dim=1)
        prediction = self.fc(final_summary)

        return prediction
def get_model(vocab_size, embed_dim=100, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
    """
    Helper function to initialize the model and move it to GPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the class
    model = LSTM(vocab_size, embed_dim,hidden_dim, output_dim)
    
    # Move to GPU
    model = model.to(device)
    
    return model, device



# Testing block
if __name__ == "__main__":
    # Mock Parameters (Must match your data loader settings roughly)
    VOCAB_SIZE = 15002
    EMBED_DIM = 100
    HIDDEN_DIM = 64
    OUTPUT_DIM = 8
    print("building model")
    model = LSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    print("model built successfully")

    dummy_input = torch.randint(0, 1000 ,(2,10))

    try:
        output = model(dummy_input)
        print(f"Input shape{dummy_input.shape}")
        print(f"Output shape: {output.shape} (should be (2,8)")

        if output.shape == (2,8) :
            print("Model forward pass success")
        else:
            print("Shape mismatch")
    except Exception as e:
        print(f"Error during forward pass{e}")