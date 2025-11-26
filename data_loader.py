import numpy as np
import pandas as pd
import os

class UniversalLoader():
    #this class is for reading, loading and merging the data from csv files

    def __init__(self, data_folder = 'data'):
        self.data_folder =data_folder
        #data paths
        self.sms_path = os.path.join(data_folder,"spam.csv")
        self.imdb_path = os.path.join(data_folder,"imdb.csv")
        self.news_train_path = os.path.join(data_folder, "news_train.csv")
        self.news_test_path = os.path.join(data_folder, "news_test.csv")
        
        #placeholders for data
        self.df_sms = None
        self.df_imdb = None
        self.df_news = None
        self.combined_df = None

    #sms loader
    def sms_loader(self) -> pd.DataFrame:
        #internal method to load & process for labels 0 & 1
        if not os.path.exists(self.sms_path):
            print(f"file path not specified {self.sms_path}")
            return pd.DataFrame()
        df = pd.read_csv(self.sms_path , encoding='latin-1')
        df = df[['v2','v1']].rename(columns={'v2':'text', 'v1':'label'})
        df['label'] = df['label'].map({'ham':0 , 'spam':1})
        return df
    
    def imdb_loader(self, limit=10000) -> pd.DataFrame:
        #internal function to load and process for labels 2 & 3
        if not os.path.exists(self.sms_path):
            print(f"file path not specified {self.sms_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.imdb_path)
        # Optimization: Limit rows for faster debugging/training
        if limit:
            df = df.head(limit)

        df = df.rename(columns={'review':'text', 'sentiment':'label'})

        df['label'] = df['label'].map({'negative':2, 'positive':3})
        return df
    
    def load_news(self, limit=10000) -> pd.DataFrame :
        # this is the internal function to load and process for news data file
        if not os.path.exists(self.news_train_path) or not os.path.exists(self.news_test_path) :
            print(f"File path not specified")
            return pd.DataFrame()
        
        df_train = pd.read_csv(self.news_train_path)
        df_test = pd.read_csv(self.news_test_path)

        df = pd.concat([df_train,df_test], axis=0).reset_index(drop=True)

        if limit:
            df = df.head(limit)

        df = df[['Description', 'Class Index']].rename(columns={'Description': 'text', 'Class Index': 'label'})
        
        # Map: Original 1-4 -> New 4-7
        df['label'] = df['label'] + 3 
        return df
    
    def combined_loader(self) -> pd.DataFrame:
        # here you combine all the files
        self.df_imdb = self.imdb_loader()
        self.df_news = self.load_news()
        self.df_sms = self.sms_loader()
        print("\n--- Merging Data ---")
        # Combine all
        self.combined_df = pd.concat([self.df_sms, self.df_imdb, self.df_news], axis=0)
        self.combined_df.dropna(inplace=True)

        self.combined_df = self.combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"✅ Successfully loaded {len(self.combined_df)} total samples.")
        print(f"   - SMS: {len(self.df_sms)}")
        print(f"   - IMDB: {len(self.df_imdb)}")
        print(f"   - News: {len(self.df_news)}")
        
        return self.combined_df

# --- USAGE BLOCK ---
if __name__ == "__main__":
    # 1. Instantiate the Class
    # Note: '.' means the current directory. Change this if your files are in a subfolder like 'data/'
    loader = UniversalLoader(data_folder='data/') 
    
    # 2. Run the loading process
    data = loader.combined_loader()
    
    # 3. Check results
    print("\nSample Rows:")
    print(data.head())