from data_loader import UniversalLoader
import pandas as pd

print("--- AUDITING RAW DATA (Before Numbers) ---")

loader = UniversalLoader(data_folder='data')
df = loader.combined_loader()

print("\n--- CHECKING SAMPLE ROWS FOR EACH LABEL ---")

# We will pick 3 examples from every label (0-7) and print them.
for label_id in range(8):
    print(f"\n🔍 CHECKING LABEL {label_id}:")
    
    # Get rows with this label
    examples = df[df['label'] == label_id]['text'].head(3).tolist()
    
    if not examples:
        print("   ❌ NO DATA FOUND FOR THIS LABEL!")
    else:
        for i, text in enumerate(examples):
            # Print first 60 chars to keep it clean
            print(f"   {i+1}. {text[:80]}...")

print("\n" + "="*50)
print("HOW TO DIAGNOSE:")
print("1. If Label 0/1 looks like News -> Your SMS Loader is broken.")
print("2. If Label 2/3 looks like News -> Your IMDB Loader is broken.")
print("3. If Label 4-7 looks like Movies -> Your News Loader is broken.")
print("="*50)