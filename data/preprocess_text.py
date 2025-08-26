import nltk
from nltk.corpus import brown
from collections import Counter
import pickle
import os
import argparse
from tqdm import tqdm

def build_vocab(sentences, min_freq=5):
    """
    Builds a word-to-index vocabulary from a list of sentences.
    
    Args:
        sentences (list): A list of tokenized sentences.
        min_freq (int): The minimum frequency for a word to be included in the vocab.
        
    Returns:
        dict: A word_to_idx mapping dictionary.
    """
    print("Building vocabulary...")
    
    # Flatten all sentences into a single list of words
    all_words = [word.lower() for sent in sentences for word in sent]
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Create the vocabulary, including only words that meet the min_freq
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create word_to_idx mapping, adding special tokens
    word_to_idx = {word: i+2 for i, word in enumerate(vocab)}
    word_to_idx['<PAD>'] = 0 # Padding token
    word_to_idx['<UNK>'] = 1 # Unknown word token
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    return word_to_idx

def sentences_to_sequences(sentences, word_to_idx):
    """
    Converts a list of tokenized sentences into sequences of integer indices.
    """
    print("Converting sentences to sequences...")
    sequences = []
    for sent in tqdm(sentences, desc="Processing sentences"):
        seq = [word_to_idx.get(word.lower(), word_to_idx['<UNK>']) for word in sent]
        sequences.append(seq)
    return sequences

def main(args):
    """
    Main function to run the preprocessing pipeline.
    """
    # Ensure the NLTK data is available
    try:
        nltk.data.find('corpora/brown')
    except nltk.downloader.DownloadError:
        print("Brown corpus not found. Please run 'get_corpus.py' first.")
        return

    # 1. Load and tokenize sentences from the corpus
    print("Loading Brown Corpus...")
    sentences = brown.sents()
    
    # 2. Build the vocabulary
    word_to_idx = build_vocab(sentences, min_freq=args.min_freq)
    
    # 3. Convert sentences to integer sequences
    sequences = sentences_to_sequences(sentences, word_to_idx)
    
    # 4. Save the processed data
    os.makedirs(args.output_dir, exist_ok=True)
    
    vocab_path = os.path.join(args.output_dir, 'word_to_idx.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(word_to_idx, f)
    print(f"Vocabulary saved to {vocab_path}")
    
    sequences_path = os.path.join(args.output_dir, 'sequences.pkl')
    with open(sequences_path, 'wb') as f:
        pickle.dump(sequences, f)
    print(f"Processed sequences saved to {sequences_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess text data for the Hierarchical CNN model.")
    
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save the processed data files.')
    parser.add_argument('--min_freq', type=int, default=5,
                        help='Minimum word frequency to be included in the vocabulary.')
                        
    args = parser.parse_args()
    main(args)
