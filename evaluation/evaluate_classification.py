
import torch
import pandas as pd
import requests
import io
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Import your model
from model.hierarchical_cnn import HierarchicalCNN

# URL for the Stanford Sentiment Treebank (SST-2) dev set from the GLUE benchmark
SST2_DEV_URL = "https://raw.githubusercontent.com/textattack/textattack/master/datasets/classification/glue/sst2/dev.tsv"

def download_sst2_data():
    """Downloads and loads the SST-2 dev set."""
    print(f"Downloading SST-2 dev set from {SST2_DEV_URL}...")
    response = requests.get(SST2_DEV_URL)
    response.raise_for_status()
    # The file is tab-separated with a header
    df = pd.read_csv(io.StringIO(response.text), sep='\t')
    return df

def generate_features(df, model, word_to_idx, device, batch_size=32):
    """Uses the Hierarchical CNN as a feature extractor."""
    model.eval()
    sentences = df['sentence'].tolist()
    labels = df['label'].values
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Generating Features"):
            batch_sentences = sentences[i:i+batch_size]
            sequences = [[word_to_idx.get(word.lower(), word_to_idx['<UNK>']) for word in sent.split()] for sent in batch_sentences]
            padded = torch.nn.utils.rnn.pad_sequence(
                [torch.LongTensor(s) for s in sequences], 
                batch_first=True, 
                padding_value=0
            ).to(device)
            
            batch_embeddings = model(padded)
            embeddings.append(batch_embeddings.cpu().numpy())
            
    return np.vstack(embeddings), labels

def main():
    # --- 1. Load Model and Vocabulary ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        with open('processed_data/word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)
    except FileNotFoundError:
        print("Vocabulary file not found. Please run data/preprocess_text.py first.")
        return
    
    model = HierarchicalCNN(
        vocab_size=len(word_to_idx),
        embedding_dim=128,
        lower_level_filters=100,
        lower_level_kernels=[2, 3, 4],
        upper_level_filters=1, upper_level_kernel=1
    ).to(device)
    
    try:
        model.load_state_dict(torch.load('models/hierarchical_cnn.pth'))
        print("Trained model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Please run unsupervised_training.py first.")
        return

    # --- 2. Load Classification Data ---
    sst2_df = download_sst2_data()
    
    # --- 3. Generate Sentence Embeddings (Features) ---
    X, y = generate_features(sst2_df, model, word_to_idx, device)
    
    # --- 4. Train and Evaluate a Linear Classifier ---
    print("\nTraining a Logistic Regression classifier on top of the embeddings...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # --- 5. Report Results ---
    predictions = classifier.predict(X_test)
    
    print("\n--- SST-2 Classification Results ---")
    print("This shows how useful the embeddings are for a downstream task.")
    print(classification_report(y_test, predictions))
    print("------------------------------------")

if __name__ == '__main__':
    main()
