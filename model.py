import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim, output_dim):
        super(LSTM,self).__init__()
        #First layer : Embed layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        #Second layer : LSTM layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        #Third layer: FUlly connected output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self,text):
        embedded = self.embedding(text)

        output, (hidden, cell) = self.lstm(embedded)

        final_memory = hidden[-1]

        prediction = self.fc(final_memory)

        return prediction
def get_model(vocab_size, embed_dim=100, hidden_dim=64, output_dim=8):
    """
    Helper function to initialize the model and move it to GPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the class
    model = LSTM(vocab_size, embed_dim, hidden_dim, output_dim)
    
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