
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import argparse
from tqdm import tqdm
from model.hierarchical_cnn import HierarchicalCNN

class SentencePairDataset(Dataset):
    """Custom Dataset for loading adjacent sentence pairs."""
    def __init__(self, sequences):
        self.pairs = []
        for i in range(len(sequences) - 1):
            self.pairs.append((sequences[i], sequences[i+1]))
            
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        sent1, sent2 = self.pairs[idx]
        return torch.LongTensor(sent1), torch.LongTensor(sent2)

def collate_fn(batch):
    """Custom collate function to pad sequences in a batch."""
    sents1, sents2 = zip(*batch)
    sents1_padded = pad_sequence(sents1, batch_first=True, padding_value=0)
    sents2_padded = pad_sequence(sents2, batch_first=True, padding_value=0)
    return sents1_padded, sents2_padded

def contrastive_loss(s_i, s_j):
    """
    Calculates the contrastive loss.
    Encourages the similarity of correct (i, j) pairs (where i=j) to be high
    and incorrect pairs (where i!=j) to be low.
    """
    # Calculate cosine similarity between all pairs of sentences in the batch
    s_i_norm = s_i / s_i.norm(dim=1)[:, None]
    s_j_norm = s_j / s_j.norm(dim=1)[:, None]
    similarity_matrix = torch.mm(s_i_norm, s_j_norm.transpose(0, 1))
    
    # The labels are the diagonal elements (i.e., the correct pairs)
    labels = torch.arange(s_i.size(0)).to(s_i.device)
    
    # Use CrossEntropyLoss, which combines LogSoftmax and NLLLoss
    # It will try to maximize the scores on the diagonal of the similarity matrix
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(similarity_matrix, labels)
    return loss

def main(args):
    # --- 1. Load Data ---
    print("Loading preprocessed data...")
    with open(f'{args.data_dir}/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    with open(f'{args.data_dir}/sequences.pkl', 'rb') as f:
        sequences = pickle.load(f)
        
    dataset = SentencePairDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # --- 2. Initialize Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HierarchicalCNN(
        vocab_size=len(word_to_idx),
        embedding_dim=args.embedding_dim,
        lower_level_filters=100,
        lower_level_kernels=[2, 3, 4],
        upper_level_filters=1, # Not used in this simplified final stage
        upper_level_kernel=1   # Not used
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # --- 3. Training Loop ---
    print("Starting unsupervised training...")
    for epoch in range(args.epochs):
        total_loss = 0
        for sents1, sents2 in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            sents1, sents2 = sents1.to(device), sents2.to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings for both sets of sentences using the same model
            emb1 = model(sents1)
            emb2 = model(sents2)
            
            # Calculate loss
            loss = contrastive_loss(emb1, emb2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    # --- 4. Save Model ---
    torch.save(model.state_dict(), args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unsupervised training of Hierarchical CNN.")
    parser.add_argument('--data_dir', type=str, default='processed_data')
    parser.add_argument('--save_path', type=str, default='models/hierarchical_cnn.pth')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
