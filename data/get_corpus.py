
import nltk
from nltk.corpus import brown

def download_corpus():
    """
    Downloads the NLTK Brown Corpus and the Punkt sentence tokenizer.
    """
    print("--- Downloading NLTK resources ---")
    try:
        # Download the Brown Corpus
        print("Downloading 'brown' corpus...")
        nltk.download('brown')
        print("✅ 'brown' corpus downloaded successfully.")
        
        # Download the Punkt tokenizer for sentence splitting
        print("\nDownloading 'punkt' tokenizer...")
        nltk.download('punkt')
        print("✅ 'punkt' tokenizer downloaded successfully.")
        
    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("Please check your internet connection and NLTK setup.")
        return

    # --- Verification Step ---
    print("\n--- Verifying download ---")
    try:
        # Load the corpus and print a sample to confirm it's available
        sentences = brown.sents()
        print(f"Successfully loaded the Brown Corpus with {len(sentences)} sentences.")
        print("Sample sentences:")
        for i in range(3):
            print(f"  - {' '.join(sentences[i])}")
    except Exception as e:
        print(f"Could not verify the corpus. Error: {e}")


if __name__ == '__main__':
    download_corpus()
