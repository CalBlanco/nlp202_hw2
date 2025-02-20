import torch

from config import START_TAG, STOP_TAG

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def make_vocab(input_data):
    word_vocab = {}
    tag_vocab = {START_TAG: 0, STOP_TAG: 1}

    for sentence, tags in input_data:
        for word in sentence:
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
        for tag in tags:
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)

    return word_vocab, tag_vocab
