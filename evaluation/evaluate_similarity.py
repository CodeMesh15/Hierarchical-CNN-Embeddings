
import torch
import pandas as pd
import requests
import io
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import numpy as np

# Import your model
from model.hierarchical_cnn import HierarchicalCNN

# URL for a preprocessed version of the STS-B development set
STS_B_DEV_URL = "https://raw.githubusercontent.com/Philip-May/stsb-multi-mt/main/data/stsb_multi_mt/stsb-en-dev.csv"

def download_sts_b_data():
    """Downloads and loads the STS-B dev set into a pandas DataFrame."""
    print(f"Downloading STS-B dev set from {STS_B_DEV_URL}...")
    response = requests.get(STS_B_DEV_URL)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    # Normalize the scores from 0-5 to 0-1 for consistency
    df['score'] = df['score'] / 5.0
    return df

def get_embeddings(sentences, model, word_to_idx, device, batch_size=32):
    """Generates embeddings for a list of sentences."""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Generating Embeddings"):
            batch_sentences = sentences[i:i+batch_size]
            
            # Convert sentences to sequences
            sequences = [[word_to_idx.get(word.lower(), word_to_idx['<UNK>']) for word in sent.split()] for sent in batch_sentences]
            
            # Pad sequences
            padded = torch.nn.utils.rnn.pad_sequence(
                [torch.LongTensor(s) for s in sequences], 
                batch_first=True, 
                padding_value=0
            ).to(device)
            
            embeddings = model(padded)
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)


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
    
    # Initialize the model with the same architecture as in training
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

    # --- 2. Load Evaluation Data ---
    sts_df = download_sts_b_data()
    
    # --- 3. Generate Embeddings ---
    # Get all unique sentences to avoid re-computing
    sents1 = sts_df['sentence1'].tolist()
    sents2 = sts_df['sentence2'].tolist()
    unique_sentences = list(set(sents1 + sents2))
    
    # Create embeddings for all unique sentences
    embeddings = get_embeddings(unique_sentences, model, word_to_idx, device)
    
    # Create a mapping from sentence to its embedding
    sent_to_emb = {sent: emb for sent, emb in zip(unique_sentences, embeddings)}
    
    # --- 4. Calculate Cosine Similarities ---
    print("Calculating cosine similarities...")
    predicted_scores = []
    for s1, s2 in zip(sents1, sents2):
        emb1 = sent_to_emb[s1].reshape(1, -1)
        emb2 = sent_to_emb[s2].reshape(1, -1)
        sim = cosine_similarity(emb1, emb2)[0, 0]
        predicted_scores.append(sim)
        
    # --- 5. Evaluate Correlation ---
    ground_truth_scores = sts_df['score'].tolist()
    
    pearson_corr, _ = pearsonr(predicted_scores, ground_truth_scores)
    spearman_corr, _ = spearmanr(predicted_scores, ground_truth_scores)
    
    print("\n--- STS-B Evaluation Results ---")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    print("---------------------------------")


if __name__ == '__main__':
    main()
