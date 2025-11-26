import pandas as pd

news_train1 = pd.read_csv(r'C:\Text_Classification\data\news_train.csv')

news_test1 = pd.read_csv(r'C:\Text_Classification\data\news_test.csv')

df_news = pd.concat([news_train1 ,news_test1], axis=0)

df_news = df_news.reset_index(drop=True)


import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

# The big test
if torch.cuda.is_available():
    print(f"SUCCESS: CUDA is available! (Using {torch.cuda.device_count()} GPU(s))")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Test a quick calculation on the GPU
    x = torch.rand(5, 5).cuda()
    print("GPU Tensor created successfully:")
    print(x)
else:
    print("WARNING: CUDA is NOT available. You are running on CPU.")
    print("Did you install the specific '--index-url' version of torch?")