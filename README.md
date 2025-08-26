# Hierarchical-CNN-Embeddings

An implementation of a Deep Hierarchical Convolutional Neural Network (CNN) for learning unsupervised semantic textual embeddings. This approach is designed to capture both local and global textual information, inspired by research performed at EPFL.

---

## 1. Project Overview

This project explores the creation of rich, semantic sentence embeddings without relying on labeled data. The core of this work is the implementation of a **Deep Hierarchical Convolutional Neural Network**. Standard CNNs are great at capturing local patterns (like n-grams), but this hierarchical approach is designed to learn global textual information by building up representations from smaller phrases to the full sentence. The goal is to produce embeddings that perform competitively on downstream tasks like sentiment analysis and semantic similarity matching.

---

## 2. Core Objectives

-   To implement a **Hierarchical CNN** architecture for processing text.
-   To train this model in an **unsupervised** manner to learn sentence embeddings.
-   To evaluate the quality of the learned embeddings on both **supervised** (sentiment classification) and **unsupervised** (semantic similarity) tasks.

---

## 3. Methodology

#### Phase 1: Data and Preprocessing

1.  **Dataset**: For unsupervised training, we need a large corpus of text. A standard choice is the text from **Wikipedia** or a book corpus like the **Brown Corpus**.
2.  **Preprocessing**:
    -   Create a vocabulary from the text corpus.
    -   Convert sentences into sequences of integer indices based on the vocabulary.
    -   Pad all sequences to a uniform length.

#### Phase 2: Model Architecture (Hierarchical CNN)

The model will be built in PyTorch or TensorFlow and will have the following structure:
1.  **Embedding Layer**: An initial layer that converts word indices into dense vectors.
2.  **Lower-Level CNN**: A set of parallel convolutional layers with small kernel sizes (e.g., 2, 3, 4). Each kernel slides over the word embeddings to capture local phrase information (bi-grams, tri-grams, etc.). This is followed by a pooling operation.
3.  **Higher-Level CNN**: The outputs from the lower-level CNNs are concatenated and treated as a new sequence. Another convolutional layer with a larger kernel size is applied to capture the relationships between the phrases, thus learning more global sentence structure.
4.  **Final Pooling**: A final pooling layer (e.g., max-pooling) is applied to the output of the higher-level CNN to produce a single, fixed-size vector representing the entire sentence.

#### Phase 3: Unsupervised Training

The key is to train this model without labels. We will use a **Skip-Thought** or **Quick-Thought** objective:
1.  **Training Goal**: Given the embedding of a sentence `s_i`, the model should be able to distinguish the embedding of the next sentence `s_{i+1}` from the embeddings of other random sentences in the same batch.
2.  **Loss Function**: We will use a contrastive loss function. The model will try to maximize the similarity (e.g., dot product) between the embeddings of adjacent sentences (`s_i` and `s_{i+1}`) while minimizing their similarity to other sentences in the batch.

#### Phase 4: Evaluation

To test how well our unsupervised model learned semantic representations, we evaluate the embeddings on downstream tasks:
1.  **Unsupervised Task (Semantic Similarity)**:
    -   Use a standard dataset like **STS-B (Semantic Textual Similarity Benchmark)**.
    -   For each pair of sentences, generate their embeddings using our trained Hierarchical CNN.
    -   Calculate the cosine similarity between the two embeddings and measure the correlation with the human-annotated scores.
2.  **Supervised Task (Sentiment Analysis)**:
    -   Use a dataset like **SST-2 (Stanford Sentiment Treebank)**.
    -   "Freeze" our trained Hierarchical CNN and use it only as a feature extractor.
    -   Train a simple linear classifier (e.g., Logistic Regression) on top of these fixed embeddings to predict sentiment. High accuracy indicates high-quality embeddings.

