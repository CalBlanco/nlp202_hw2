import torch

from config import START_TAG, STOP_TAG, UNKNOWN_TAG, PAD_TAG, device


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    """Convert words to indices"""
    idxs = [to_ix.get(w, to_ix.get(UNKNOWN_TAG, 0)) for w in seq]
    return torch.tensor(idxs, dtype=torch.long, device=device)

def prepare_char_sequence(seq, to_ix):
    """Convert characters to indices"""
    idxs = [[to_ix.get(c, to_ix.get(UNKNOWN_TAG, 0)) for c in word] for word in seq]
    return torch.tensor(idxs, dtype=torch.long, device=device)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def make_vocab(input_data, character=False):
    """Create vocabulary from data"""
    word_vocab = {UNKNOWN_TAG: 0, PAD_TAG: 1}
    tag_vocab = {PAD_TAG: 0, START_TAG: 1, STOP_TAG: 2}

    for sentence, tags in input_data:
        for word in sentence:
            if character:
                for char in word:
                    if char not in word_vocab:
                        word_vocab[char] = len(word_vocab)
                continue

            else:
                if word not in word_vocab:
                    word_vocab[word] = len(word_vocab)
        for tag in tags:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)

    return word_vocab, tag_vocab

def log_sum_exp_batch(vec):
    # vec: [batch_size, tagset_size, tagset_size]
    max_score = vec.max(2)[0]  # [batch_size, tagset_size]
    max_score_broadcast = max_score.unsqueeze(-1)  # [batch_size, tagset_size, 1]
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=2))

def prepare_sequence_batch(batch_seqs, to_ix):
    """Convert batch of sequences to padded tensor"""
    max_len = max(len(seq) for seq in batch_seqs)
    batch_tensor = torch.zeros((len(batch_seqs), max_len), dtype=torch.long, device=device)
    
    for i, seq in enumerate(batch_seqs):
        idxs = [to_ix.get(w, to_ix.get(UNKNOWN_TAG, 0)) for w in seq]
        batch_tensor[i, :len(idxs)] = torch.tensor(idxs, dtype=torch.long, device=device)
    
    return batch_tensor

def load_data(file_path):
    """Load data from file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                sentence = parts[0].split()
                tags = parts[1].split()
                if len(sentence) == len(tags):
                    data.append((sentence, tags))
    return data
