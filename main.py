import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import json
import sys

from models import BiLSTM_CRF
from util import make_vocab, load_data, prepare_sequence, prepare_sequence_batch
from config import device, PAD_TAG

# Model configuration
MODEL_CONFIG = {
    'use_char_cnn': True,  # Enable/disable character CNN
    'use_pos': False,       # Enable/disable POS features
    'use_caps': False,      # Enable/disable capitalization features
    'loss_fn': 'ramp',       # Options: 'crf', 'softmax_margin', 'ramp'
}

EMBEDDING_DIM = 4
HIDDEN_DIM = 4
CHAR_EMBEDDING_DIM = 30
NUM_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else 'predictions.txt'

class LossFunctions:
    @staticmethod
    def softmax_margin_loss(logits, true_tags, mask=None, margin=1.0):
        """
        Implements Softmax Margin Loss for sequence labeling
        Args:
            logits: Shape [batch_size, seq_len, num_tags]
            true_tags: Shape [batch_size, seq_len]
            mask: Shape [batch_size, seq_len]
            margin: Margin for incorrect predictions
        """
        batch_size, seq_len, num_tags = logits.size()
        
        # Convert true_tags to one-hot
        true_tags_onehot = F.one_hot(true_tags, num_tags).float()
        
        # Compute softmax scores
        scores = F.log_softmax(logits, dim=2)
        
        # Add margin to all incorrect tags
        margin_tensor = margin * (1 - true_tags_onehot)
        margin_scores = scores - margin_tensor
        
        # Compute loss
        loss = -torch.sum(true_tags_onehot * margin_scores, dim=2)
        
        if mask is not None:
            loss = loss * mask
            
        return loss.sum() / (mask.sum() if mask is not None else batch_size * seq_len)

    @staticmethod
    def ramp_loss(logits, true_tags, mask=None, margin=1.0, gamma=1.0):
        """
        Implements Ramp Loss for sequence labeling
        Args:
            logits: Shape [batch_size, seq_len, num_tags]
            true_tags: Shape [batch_size, seq_len]
            mask: Shape [batch_size, seq_len]
            margin: Margin parameter
            gamma: Scaling parameter
        """
        batch_size, seq_len, num_tags = logits.size()
        
        # Convert true_tags to one-hot
        true_tags_onehot = F.one_hot(true_tags, num_tags).float()
        
        # Compute scores for true tags
        true_scores = torch.sum(logits * true_tags_onehot, dim=2)
        
        # Compute max scores for incorrect tags
        masked_logits = logits - true_tags_onehot * 1e9  # Mask out true tags
        max_wrong_scores = torch.max(masked_logits, dim=2)[0]
        
        # Compute ramp loss
        loss = F.relu(margin - (true_scores - max_wrong_scores) / gamma)
        
        if mask is not None:
            loss = loss * mask
            
        return loss.sum() / (mask.sum() if mask is not None else batch_size * seq_len)

def get_loss_function(loss_type):
    if loss_type == 'crf':
        return None  # CRF loss is handled in the model
    elif loss_type == 'softmax_margin':
        return LossFunctions.softmax_margin_loss
    elif loss_type == 'ramp':
        return LossFunctions.ramp_loss
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")

# Get the selected loss function
loss_fn = get_loss_function(MODEL_CONFIG['loss_fn'])

# Optional feature configurations
char_config = None
if MODEL_CONFIG['use_char_cnn']:
    char_config = {
        'char_embedding_dim': CHAR_EMBEDDING_DIM,
        'cnn_filters': [(2, 25), (3, 25), (4, 25)]
    }

def build_char_vocab(data):
    if not MODEL_CONFIG['use_char_cnn']:
        return None
    char_vocab = {PAD_TAG: 0}
    for sentence, _ in data:
        for word in sentence:
            for char in word:
                if char not in char_vocab:
                    char_vocab[char] = len(char_vocab)
    return char_vocab

def prepare_char_sequence_batch(sentences, char_vocab, max_word_len=20):
    if not MODEL_CONFIG['use_char_cnn'] or char_vocab is None:
        return None
    batch_size = len(sentences)
    max_seq_len = max(len(s) for s in sentences)
    char_ids = torch.zeros((batch_size, max_seq_len, max_word_len), dtype=torch.long)
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            char_sequence = [char_vocab.get(c, char_vocab[PAD_TAG]) for c in word[:max_word_len]]
            char_ids[i, j, :len(char_sequence)] = torch.tensor(char_sequence)
    return char_ids

class NERDataset(Dataset):
    def __init__(self, data, word_vocab, tag_vocab, char_vocab=None):
        self.data = data
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab
        self.use_char_cnn = MODEL_CONFIG['use_char_cnn']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = self.data[idx]
        return sentence, tags

def collate_fn(batch, word_vocab, tag_vocab, char_vocab=None):
    sentences, tags = zip(*batch)
    features = {
        'words': prepare_sequence_batch(sentences, word_vocab),
        'tags': prepare_sequence_batch(tags, tag_vocab)
    }
    
    if MODEL_CONFIG['use_char_cnn'] and char_vocab is not None:
        features['chars'] = prepare_char_sequence_batch(sentences, char_vocab)
        
    return features

# Load and prepare data
train_data = load_data(os.path.join(os.path.dirname(__file__), 'A2-data/train'))
val_data = load_data(os.path.join(os.path.dirname(__file__), 'A2-data/dev.answers'))
test_data = load_data(os.path.join(os.path.dirname(__file__), 'A2-data/test.answers'))

TRAIN_SIZE = 0.5
train_chunk = int(len(train_data) * TRAIN_SIZE)
train_data = train_data[:train_chunk]

# Build vocabularies
words_vocab, tags_vocab = make_vocab(train_data)
char_vocab = build_char_vocab(train_data) if MODEL_CONFIG['use_char_cnn'] else None

if MODEL_CONFIG['use_char_cnn']:
    char_config['char_vocab_size'] = len(char_vocab)

# Initialize model with optional features
model = BiLSTM_CRF(
    vocab_size=len(words_vocab),
    tag_to_ix=tags_vocab,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    char_config=char_config,
    use_crf=MODEL_CONFIG['loss_fn'] == 'crf'
).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Data preparation with optional features
train_dataset = NERDataset(train_data, words_vocab, tags_vocab, char_vocab)
val_dataset = NERDataset(val_data, words_vocab, tags_vocab, char_vocab)
test_dataset = NERDataset(test_data, words_vocab, tags_vocab, char_vocab)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda x: collate_fn(x, words_vocab, tags_vocab, char_vocab)
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda x: collate_fn(x, words_vocab, tags_vocab, char_vocab)
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda x: collate_fn(x, words_vocab, tags_vocab, char_vocab)
)

# Training loop
best_val_loss = float('inf')
best_model = None

for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    
    for batch in tqdm(train_loader, desc=f'Epoch {epoch} - Training'):
        # Move all features to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        model.zero_grad()
        loss = model(
            batch['words'],
            char_x=batch.get('chars'),
            tags=batch['tags'],
            mask=None,  # Add mask if needed
            loss_fn=loss_fn
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch} - Validation'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            loss = model(
                batch['words'],
                char_x=batch.get('chars'),
                tags=batch['tags'],
                mask=None,  # Add mask if needed
                loss_fn=loss_fn
            )
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model = model.state_dict().copy()
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")

# Testing
if best_model:
    model.load_state_dict(best_model)

model.eval()
results = []
with torch.no_grad():
    batch_idx = 0
    for batch in test_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        batch_predictions = model(
            batch['words'],
            char_x=batch.get('chars')
        )
        
        for i, predictions in enumerate(batch_predictions):
            original_sentence = test_data[batch_idx + i][0]
            actual_length = len(original_sentence)
            predictions = predictions[:actual_length]
            decoded_tags = [list(tags_vocab.keys())[list(tags_vocab.values()).index(t)] for t in predictions]
            results.append((original_sentence, decoded_tags))
        
        batch_idx += len(batch['words'])

with open(OUTFILE, 'w') as f:
    for original_sentence, decoded_tags in results:
        for i, word in enumerate(original_sentence):
            f.write(f'{word}\t{decoded_tags[i]}\n')
        f.write('\n')
