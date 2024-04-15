

import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import spacy

def tokenize(text):
    nlp = spacy.load("en_core_web_sm")
    return [token.text for token in nlp(text)]

def generate_vocabulary(train_data):
    counter = Counter()
    for example in train_data['text']:
        counter.update(tokenize(example))
    vocab = sorted(counter, key=counter.get, reverse=True)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    return word_to_idx

def process_data(data, word_to_idx):
    texts = []
    labels = []
    for text, label in zip(data['text'], data['label']):
        tokens = tokenize(text)
        text_indices = [word_to_idx.get(token, 0) for token in tokens]  # 0 for unknown tokens
        texts.append(torch.LongTensor(text_indices))
        labels.append(label)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts, labels