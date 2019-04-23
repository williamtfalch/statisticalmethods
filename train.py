import json
import re
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch
from copy import copy
from torch.autograd import Variable
from torch.nn import functional as F
import sys

w2v = {}
labels = None
premises = None
hypotheses = None
batch_size = 64

original_representation_lengths = []
longest_representation = 0


class Unidirectional(nn.Module):
    def __init__(self):
        super(Unidirectional, self).__init__()
        self.lstm_premise = nn.LSTM(300, 2048, 1)
        self.lstm_hypothesis = nn.LSTM(300, 2048, 1)

    def forward(self, x1, x2):
        x1 = self.lstm_premise(x1)
        x2 = self.lstm_hypothesis(x2)
        x = torch.mul((x1, x2))
        x = nn.Linear(300, 512)
        x = nn.Linear(512, 3)
        x = nn.Softmax()

        return x


class Bidirectional(nn.Module):
    def __init__(self):
        super(Bidirectional, self).__init__()
        self.lstm_premise = nn.LSTM(300, 2048, 1)
        self.lstm_hypothesis = nn.LSTM(300, 2048, 1)

    def forward(self, x1, x2):
        x1 = self.lstm_premise(x1)
        x1 = self.lstm_premise(x1[::-1])
        x2 = self.lstm_hypothesis(x2)
        x2 = self.lstm_premise(x2[::-1])
        x = torch.cat((x1, x2))
        x = nn.Linear(300 * 2, 512)
        x = nn.Linear(512, 3)
        x = nn.Softmax()

        return x


class BidirectionalMaxPool(nn.Module):
    def __init__(self):
        super(BidirectionalMaxPool, self).__init__()
        self.lstm_premise = nn.LSTM(300, 2048, 1)
        self.lstm_hypothesis = nn.LSTM(300, 2048, 1)
        self.pool_factor = 3

    def forward(self, x1, x2):
        x1 = self.lstm_premise(x1)
        x1 = self.lstm_premise(x1[::-1])
        x2 = self.lstm_hypothesis(x2)
        x2 = self.lstm_premise(x2[::-1])
        x = torch.cat((x1, x2))
        x = nn.MaxPool1d(self.pool_factor),
        x = nn.Linear(300 * 2 / self.pool_factor, 512)
        x = nn.Linear(512, 3)
        x = nn.Softmax()

        return x

#### Support functions ####


def load_data(path, cutoff=False, to_json=False):
    ticker = 0
    data = []

    with open(path) as f:
        for line in f:

            if to_json:
                line = json.loads(line)

            data.append(line)

            ticker += 1

            if cutoff and ticker >= cutoff:
                f.close()
                break

    return data


def process_glove_embeddings(num_words=1000):
    glove_path = "./glove.840B.300d.txt"
    words = {}

    data = load_data(glove_path, num_words)

    for d in data:
        word_index = d.find(" ")
        word = d[:word_index]

        embedding = d[word_index + 1:].split(" ")

        words[word] = embedding

    return words


def sentence_to_list(sentence):
    return re.findall(r"[A-Za-z@#]+|\S", sentence)


def load_snli_corpus(path, num_examples=1000, to_numbers=False):
    global w2v
    ticker = 0

    snli_corpus = load_data(path, num_examples, True)

    labels = [e['annotator_labels'][0] for e in snli_corpus]
    premises = [sentence_to_list(e['sentence1']) for e in snli_corpus]
    hypotheses = [sentence_to_list(e['sentence2']) for e in snli_corpus]

    for l in range(len(labels)):
        if labels[l] == "-":
            del labels[l]
            del premises[l]
            del hypotheses[l]

            l -= 1

        elif labels[l] == "entailment":
            labels[l] = [1, 0, 0]

        elif labels[l] == "neutral":
            labels[l] = [0, 1, 0]

        elif labels[l] == "contradiction":
            labels[l] = [0, 0, 1]

        for word in premises[l]:
            if word not in w2v:
                w2v[word] = ticker
                ticker += 1

        for word in hypotheses[l]:
            if word not in w2v:
                w2v[word] = ticker
                ticker += 1

    return labels, premises, hypotheses


def extract_relations_multiply(a, b):
    return a*b


def extract_relations_diff(a, b):
    return abs(a-b)


def extract_relations_concat(a, b):
    return a+b

#### end Support functions ####

#### Representations ####


def get_baseline_representation(path, num_examples=10000):
    glove_path = "./glove.840B.300d.txt"
    labels, premises, hypotheses = load_snli_corpus(path, num_examples)

    glove_embeddings = {}
    premises_embeddings = []
    hypotheses_embeddings = []
    premise_embedding = None
    hypotheses_embedding = None
    readlines = 0

    with open(glove_path) as f:
        for i in range(len(labels)):
            premise_embedding = []
            hypotheses_embedding = []

            for word in premises[i]:
                while word not in glove_embeddings:
                    glove_embedding = f.readline()
                    readlines += 1

                    word_index = glove_embedding.find(" ")
                    word = glove_embedding[:word_index]
                    embedding = list(
                        map(float, glove_embedding[word_index + 1:].split(" ")))

                    glove_embeddings[word] = embedding

                premise_embedding.append(glove_embeddings[word])

            premise_embedding = [sum(x)/len(premise_embedding)
                                 for x in zip(*premise_embedding)]

            premises_embeddings.append(premise_embedding)

            for word in hypotheses[i]:
                while word not in glove_embeddings:
                    glove_embedding = f.readline()
                    readlines += 1

                    word_index = glove_embedding.find(" ")
                    word = glove_embedding[:word_index]
                    embedding = list(
                        map(float, glove_embedding[word_index + 1:].split(" ")))

                    glove_embeddings[word] = embedding

                hypotheses_embedding.append(glove_embeddings[word])

            hypotheses_embedding = [sum(x)/len(hypotheses_embedding)
                                    for x in zip(*hypotheses_embedding)]

            hypotheses_embeddings.append(hypotheses_embedding)

        f.close()

    premises_hypotheses_relations = extract_relations(
        premises_embeddings, hypotheses_embeddings)

    return (labels, premises_hypotheses_relations)


def get_lstm_representation(path, num_examples=10000):
    global w2v
    global longest_representation
    global original_representation_lengths

    pad_token = 0

    labels, premises, hypotheses = load_snli_corpus(path, num_examples)

    longest_representation = max(
        len(max(premises, key=len)), len(max(hypotheses, key=len)))

    for i in range(len(premises)):
        len_p = len(premises[i])
        encoded_p = [pad_token] * longest_representation
        encoded_p[:len_p] = [w2v[premises[i][j]]
                             for j in range(len_p)]
        premises[i] = encoded_p

        len_h = len(hypotheses[i])
        encoded_h = [pad_token] * longest_representation
        encoded_h[:len_h] = [w2v[hypotheses[i][k]]
                             for k in range(len_h)]
        hypotheses[i] = encoded_h

        original_representation_lengths.append((len_p, len_h))

    return (labels, premises, hypotheses)


#### end Representations ####


def get_corpus_representation(encoder, path, num_examples=10000):
    if encoder == "baseline":
        return get_baseline_representation(path, num_examples)

    else:
        return get_lstm_representation(path, num_examples)


def extract_relations(premises, hypotheses, method="multiply"):
    relations = []

    for i in range(len(premises)):
        result = None

        if method == "multiply":
            result = [extract_relations_multiply(
                a, b) for a, b in zip(premises[i], hypotheses[i])]

        elif method == "diff":
            result = [extract_relations_diff(
                a, b) for a, b in zip(premises[i], hypotheses[i])]

        else:
            result = [extract_relations_concat(
                a, b) for a, b in zip(premises[i], hypotheses[i])]

        relations.append(result)

    return relations


def get_model(encoder):
    n_in, n_h, n_out = 300, 512, 3

    if encoder == "baseline":
        return nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.Linear(n_h, n_out),
            nn.Softmax()
        )

    elif encoder == "unidirectional":
        return Unidirectional()

    elif encoder == "bidirectional":
        return Bidirectional()

    elif encoder == "bidirectionalmaxpool":
        return BidirectionalMaxPool()


def main():
    arguments = sys.argv[1:]
    encoder = None
    path = None

    if len(arguments) > 0:
        encoder = arguments[0]

    else:
        encoder = "baseline"

    if len(arguments) > 1:
        path = arguments[1]

    else:
        path = "./snli_1.0/snli_1.0_train.jsonl"

    data = get_corpus_representation(encoder, path)
    model = get_model(encoder)


if __name__ == "__main__":
    main()
