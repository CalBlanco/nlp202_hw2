import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import json
import os

from models import BiLSTM_CRF
from util import make_vocab, load_data, prepare_sequence
from config import device, setup_device

torch.manual_seed(1)
torch.device(device)

EMBEDDING_DIM = 4
HIDDEN_DIM = 4
train_data = load_data('A2-Data/train')

words_vocab, tags_vocab = make_vocab(train_data)

model = BiLSTM_CRF(len(words_vocab), tags_vocab, EMBEDDING_DIM, HIDDEN_DIM).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(train_data[0][0], words_vocab)
    precheck_tags = torch.tensor([tags_vocab[t] for t in train_data[0][1]], dtype=torch.long, device=device)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in tqdm(range(20), desc='Training'):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, words_vocab)
        targets = torch.tensor([tags_vocab[t] for t in tags], dtype=torch.long, device=device)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(train_data[0][0], words_vocab)
    print(model(precheck_sent))