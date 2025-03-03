import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=100, hidden_dim=128, char_config=None, use_crf=True):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.use_crf = use_crf
        
        # Word embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # Character CNN configuration
        self.use_char_cnn = char_config is not None
        if self.use_char_cnn:
            self.char_embedding_dim = char_config.get('char_embedding_dim', 30)
            self.char_vocab_size = char_config.get('char_vocab_size')
            self.char_cnn_filters = char_config.get('cnn_filters', [(2, 25), (3, 25), (4, 25)])
            
            # Character embedding layer
            self.char_embedding = nn.Embedding(self.char_vocab_size, self.char_embedding_dim)
            
            # Character CNN layers
            self.char_convs = nn.ModuleList([
                nn.Conv1d(self.char_embedding_dim, n_filters, kernel_size)
                for kernel_size, n_filters in self.char_cnn_filters
            ])
            
            total_filters = sum(f[1] for f in self.char_cnn_filters)
            lstm_input_dim = embedding_dim + total_filters
        else:
            lstm_input_dim = embedding_dim
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Linear layer to map to tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF layer (optional)
        if self.use_crf:
            self.crf = CRF(self.tagset_size, batch_first=True)
        
    def _get_char_cnn_features(self, char_x):
        if not self.use_char_cnn:
            return None
            
        # char_x: [batch_size, seq_len, char_seq_len]
        batch_size, seq_len, char_seq_len = char_x.size()
        
        # Reshape and embed characters
        char_x = char_x.view(-1, char_seq_len)  # [batch_size * seq_len, char_seq_len]
        char_embeds = self.char_embedding(char_x)  # [batch_size * seq_len, char_seq_len, char_emb_dim]
        char_embeds = char_embeds.transpose(1, 2)  # [batch_size * seq_len, char_emb_dim, char_seq_len]
        
        # Apply CNN and max-pooling
        conv_outputs = []
        for conv in self.char_convs:
            conv_out = F.relu(conv(char_embeds))  # [batch_size * seq_len, n_filters, conv_out_len]
            pool_out = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size * seq_len, n_filters, 1]
            conv_outputs.append(pool_out.squeeze(2))
        
        char_features = torch.cat(conv_outputs, dim=1)  # [batch_size * seq_len, total_filters]
        return char_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, total_filters]
    
    def _get_lstm_features(self, x, char_x=None):
        word_embeds = self.word_embeds(x)  # [batch_size, seq_len, embedding_dim]
        
        if self.use_char_cnn and char_x is not None:
            char_features = self._get_char_cnn_features(char_x)
            combined_embeds = torch.cat([word_embeds, char_features], dim=2)
        else:
            combined_embeds = word_embeds
            
        lstm_out, _ = self.lstm(combined_embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def forward(self, x, char_x=None, tags=None, mask=None, loss_fn=None):
        # Get BiLSTM features
        lstm_feats = self._get_lstm_features(x, char_x)
        
        # If using CRF
        if self.use_crf:
            if tags is not None:
                return -self.crf(lstm_feats, tags, mask=mask)
            return self.crf.decode(lstm_feats, mask=mask)
        
        # If using alternative loss functions
        if tags is not None and loss_fn is not None:
            return loss_fn(lstm_feats, tags, mask=mask)
        
        # For inference without CRF
        return torch.argmax(lstm_feats, dim=2)

# Training function
def train_model(model, train_data, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for sentences, tags, mask in train_data:
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(sentences, tags, mask)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data)}")

# Inference function
def predict(model, sentence, word_to_ix):
    model.eval()
    with torch.no_grad():
        # Convert sentence to tensor
        idxs = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in sentence]
        tensor = torch.LongTensor([idxs])
        
        # Create mask (all 1s for this example)
        mask = torch.ones(tensor.size(), dtype=torch.bool)
        
        # Get tag sequence
        tag_seq = model(tensor, mask=mask)[0]
        return tag_seq