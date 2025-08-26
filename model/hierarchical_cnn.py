

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCNN(nn.Module):
    """
    A Hierarchical Convolutional Neural Network for learning sentence embeddings.
    """
    def __init__(self, vocab_size, embedding_dim, lower_level_filters, 
                 lower_level_kernels, upper_level_filters, upper_level_kernel, dropout=0.5):
        """
        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimension of word embeddings.
            lower_level_filters (int): Number of filters for each lower-level CNN.
            lower_level_kernels (list of int): List of kernel sizes (n-grams) for lower-level CNNs.
            upper_level_filters (int): Number of filters for the upper-level CNN.
            upper_level_kernel (int): Kernel size for the upper-level CNN.
            dropout (float): Dropout probability.
        """
        super(HierarchicalCNN, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Lower-Level CNNs (for phrase representation)
        # One for each kernel size (e.g., bigrams, trigrams, etc.)
        self.lower_convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=lower_level_filters, kernel_size=k)
            for k in lower_level_kernels
        ])
        
        # 3. Higher-Level CNN (for sentence representation from phrases)
        # Its input will be the concatenated outputs of the lower-level CNNs
        num_lower_level_features = lower_level_filters * len(lower_level_kernels)
        self.upper_conv = nn.Conv1d(
            in_channels=num_lower_level_features, 
            out_channels=upper_level_filters, 
            kernel_size=upper_level_kernel
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        """
        # x shape: (batch_size, seq_len)
        
        # 1. Pass through embedding layer
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        # Conv1d expects (batch_size, channels, seq_len). Here, channels is the embedding_dim.
        embedded = embedded.permute(0, 2, 1)
        # embedded shape: (batch_size, embedding_dim, seq_len)
        
        # 2. Lower-Level convolutions and pooling
        lower_level_outputs = []
        for conv in self.lower_convs:
            conved = F.relu(conv(embedded))
            # conved shape: (batch_size, lower_level_filters, new_seq_len)
            pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
            # pooled shape: (batch_size, lower_level_filters)
            lower_level_outputs.append(pooled)
            
        # 3. Concatenate lower-level outputs
        concatenated = torch.cat(lower_level_outputs, dim=1)
        # concatenated shape: (batch_size, lower_level_filters * num_kernels)
        
        concatenated = self.dropout(concatenated)
        
        # 4. Higher-Level convolution
        # To apply the next convolution, we need to treat the concatenated features as a new sequence
        upper_input = concatenated.unsqueeze(2)
        # upper_input shape: (batch_size, num_lower_level_features, 1)
        
        # This part of the architecture can vary. A simpler approach might be a linear layer.
        # Following the hierarchical CNN idea, we apply another conv layer.
        # Since the sequence length is now short, the upper conv acts like a fully connected layer
        # over the phrase representations.
        
        # For a meaningful convolution, let's reshape differently
        upper_input = concatenated.unsqueeze(1) # (batch_size, 1, num_lower_level_features)
        
        # For this example, let's use a simpler final stage: a linear layer.
        # This is more common and robust than a conv on a sequence of 1.
        # We'll define it here for simplicity.
        # final_fc = nn.Linear(concatenated.shape[1], final_embedding_size).to(x.device)
        # sentence_embedding = final_fc(concatenated)
        
        # The output of this stage is our final sentence embedding
        sentence_embedding = concatenated # Let's use the concatenated phrase features directly
        
        return sentence_embedding
