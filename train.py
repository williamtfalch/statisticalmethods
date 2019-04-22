import json
import re
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch
from copy import copy
from torch.autograd import Variable
from torch.nn import functional as F

w2v = {}
labels = None
premises = None
hypotheses = None
batch_size = 64

original_representation_lengths = []
longest_representation = 0

'''
class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.fc1 = nn.LSTM(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x


class MyUnidirectional(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyUnidirectional, self).__init__()
        self.lstm_premise = nn.LSTM(input_size, hidden_size, 1)
        self.lstm_hypothesis = nn.LSTM(input_size, hidden_size, 1)

    def forward(self, x1, x2):
        x1 = self.lstm_premise(x1)
        x2 = self.lstm_hypothesis(x2)
        x = torch.mul((x1, x2))
        x = nn.Linear(len(x), 512)
        x = nn.Linear(512, 3)
        x = nn.Softmax()

        return x


class mLSTM(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_lstm_units, batch_size):
        global w2v

        self.num_layers = num_layers
        self.num_lstm_units = num_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.num_tags = 3

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # build embedding layer first
        num_vocab_words = len(w2v.keys())

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = 0

        self.word_embedding = nn.Embedding(
            num_embeddings=num_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.num_lstm_units, self.num_tags)

    def init_hidden(self):
        # the weights are of the form (num_layers, batch_size, num_lstm_units)
        hidden_a = torch.randn(self.hparams.num_lstm_layers,
                               self.batch_size, self.num_lstm_units)
        hidden_b = torch.randn(self.hparams.num_lstm_layers,
                               self.batch_size, self.num_lstm_units)

        '''
   if self.hparams.on_gpu:
        hidden_a = hidden_a.cuda()
        hidden_b = hidden_b.cuda()
    '''

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(
            X, X_lengths, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, num_lstm_units) -> (batch_size * seq_len, num_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_tag(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, num_lstm_units) -> (batch_size, seq_len, num_tags)
        X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, num_tags)
        X = X.view(batch_size, seq_len, self.num_tags)

        Y_hat = X
        return Y_hat
'''

'''# Create models and load state_dicts
modelA = MyModelA()
modelB = MyModelB()
# Load state dicts
modelA.load_state_dict(torch.load(PATH))
modelB.load_state_dict(torch.load(PATH))

model = MyEnsemble(modelA, modelB)
x1, x2 = torch.randn(1, 10), torch.randn(1, 20)
output = model(x1, x2)'''

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


def load_snli_corpus(num_examples=1000, to_numbers=False):
    global w2v
    ticker = 0

    snli_base_path = "./snli_1.0/snli_1.0_"
    snli_train_path = snli_base_path + "train.jsonl"
    snli_corpus = load_data(snli_train_path, num_examples, True)

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


def get_baseline_representation(num_examples=10000):
    glove_path = "./glove.840B.300d.txt"
    labels, premises, hypotheses = load_snli_corpus(num_examples)

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


def get_lstm_representation(num_examples=10000):
    global w2v
    global longest_representation
    global original_representation_lengths

    pad_token = 0

    labels, premises, hypotheses = load_snli_corpus(num_examples)

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


def run_lstm(input):
    input_size = len(input)

    input = [[input]]
    print(input)

    input = torch.Tensor(input)

    lstm = nn.LSTM(input_size, input_size)
    h0 = torch.randn(1, input_size)
    c0 = torch.randn(1, input_size)

    output, (hn, cn) = lstm(input, (h0, c0))

    return output


def get_corpus_representation(encoder, num_examples=10000):
    if encoder == "baseline":
        return get_baseline_representation(num_examples)

    else:
        return get_lstm_representation(num_examples)


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


def get_model(model):
    n_in, n_h, n_out = 300, 512, 3

    if model == "baseline":
        return nn.Sequential(
            nn.Linear(n_in, n_h),
            # nn.ReLU(),
            nn.Linear(n_h, n_out),
            # nn.ReLU(),
            nn.Softmax()
        )

    '''
    elif model == "unidirectional":
        nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.num_lstm_units,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )
    '''


class UnidirectionalLSTM(torch.nn.Module):
    def __init__(self, lstm_in, lstm_hidden):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(UnidirectionalLSTM, self).__init__()
        self.x_premises = torch.nn.LSTM(
            input_size=lstm_in, hidden_size=lstm_hidden)
        self.x_hypotheses = torch.nn.LSTM(
            input_size=lstm_in, hidden_size=lstm_hidden)
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def train(encoder):
    lr = 0.1
    num_epochs = 100
    loss = None
    last_loss = None

    data = get_corpus_representation(encoder)
    model = get_model(encoder)

    '''
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(num_epochs):
        for batch in range(int(len(premises_and_hypotheses_relations)/batch_size) + 1):
            x = torch.FloatTensor(premises_and_hypotheses_relations[batch *
                                                                    batch_size: (batch + 1) * batch_size])
            y = torch.Tensor(
                labels[batch * batch_size: (batch + 1) * batch_size])

            y_pred = model(x)
            # Compute and print loss
            print(y_pred)
            print("--1--")
            print(y)
            print("--2--")
            loss = criterion(y_pred, y)
            print('epoch: ', epoch, ' loss: ', loss.item())
            # Zero the gradients
            optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()

            last_loss = loss.item()

        if loss.item() > last_loss:
            for g in optimizer.param_groups:
                new_lr = lr / 5

                g['lr'] = new_lr
                lr = new_lr

                if new_lr < 0.00001:
                    print('Final epoch: ', epoch, ' loss: ', loss.item())
                    return
    '''


def run(encoder="baseline"):
    train(encoder)


run("unidirectional")
