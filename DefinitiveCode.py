
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import wandb
from torch.utils.data import Subset

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

dataset = './data/spa_sample100.txt'

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(dataset, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 25

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        #print(pair)
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    
def sentenceFromIndexes(lang, indexes):
    return [ lang.index2word[ind]  for ind in indexes if (ind != SOS_token and ind != EOS_token)]
 
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloaders(batch_size, val_split=0.2):
    input_lang, output_lang, pairs = prepareData('eng', 'spa', False)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    dataset = TensorDataset(torch.LongTensor(input_ids).to(device),
                            torch.LongTensor(target_ids).to(device))

    # Calcular el tamaño de los conjuntos de entrenamiento y validación
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size

    # Crear índices para los conjuntos de entrenamiento y validación
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Crear subconjuntos de entrenamiento y validación
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Crear DataLoaders para el conjunto de entrenamiento y validación
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return input_lang, output_lang, train_dataloader, val_dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, val_dataloader):
     
    # Modo de entrenamiento
    encoder.train()
    decoder.train()

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(dataloader)

    total_val_loss = 0
    bleu_scores = []
    selected_translations = []
    encoder.eval()  # Cambiar a modo de evaluación
    decoder.eval()  # Cambiar a modo de evaluación
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            #print(idx)
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            val_loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            total_val_loss += val_loss.item()

            #print(target_tensor.size(0))
            for i in range(target_tensor.size(0)):  # Iterar sobre las secuencias en el batch
                input_sentence = sentenceFromIndexes(input_lang, input_tensor[i].tolist())
                target_sentence = sentenceFromIndexes(output_lang, target_tensor[i].tolist())
                #print(input_sentence, target_sentence)
                predicted_indexes = torch.argmax(decoder_outputs[i], dim=1).tolist()
                predicted_sentence = sentenceFromIndexes(output_lang, predicted_indexes)
                #print("Predictet sentences: ",predicted_sentence)
                    
                # Calcular BLEU y METEOR para cada frase
                bleu = sentence_bleu([target_sentence], predicted_sentence)
                bleu_scores.append(bleu)

                if i <= 10:
                    input_sentence = ' '.join(input_sentence)
                    target_sentence = ' '.join(target_sentence)
                    predicted_sentence = ' '.join(predicted_sentence)
                    dic = {"Input Sentence": input_sentence, "Target Sentence" : target_sentence, "Predicted Sentence": predicted_sentence, "Bleu Score": bleu}
                    #print(dic)
                    selected_translations.append(dic)
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)


    return avg_train_loss, avg_val_loss, avg_bleu_score, selected_translations


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(train_dataloader, val_dataloader , encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    plot_val_losses = []
    print_loss_total = 0  # Reset every print_every
    print_val_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    plot_val_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    selected_indices = [2,4,6,8,10]

    translations_per_epoch = []

    for epoch in range(1, n_epochs + 1):
        train_loss, val_loss, avg_bleu_score, translations = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, val_dataloader)
        print_loss_total += train_loss
        print_val_loss_total += val_loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
        
            print_val_loss_avg = print_val_loss_total / print_every
            print_val_loss_total = 0
            wandb.log({"Validation Loss": print_val_loss_avg,"Training loss": print_loss_avg, "Bleu Score": avg_bleu_score}, step = epoch)
            print('%s (%d %d%%) Train Loss: %.4f, Val Loss: %.4f' % (timeSince(start, epoch / n_epochs),
                                                                         epoch, epoch / n_epochs * 100, print_loss_avg, print_val_loss_avg))
            
            epoch_translations = {
                "Epoch": epoch,
                "Sentences": translations
            }
            translations_per_epoch.append(epoch_translations)

            # Guardar las traducciones en un archivo JSON
            with open('translations_hidden.json', 'w') as json_file:
                json.dump(translations_per_epoch, json_file, ensure_ascii=False, indent=4)
  
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            plot_val_loss_avg = plot_val_loss_total / plot_every
            plot_val_losses.append(plot_val_loss_avg)
            plot_val_loss_total = 0

    

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


hidden_size = 512
batch_size = 500
epoch = 50
learning_rate = 0.0001

input_lang, output_lang, train_dataloader, val_dataloader = get_dataloaders(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

wandb.init(project="Machine Translation", config={
        									"epochs": epoch, 
                                            "learning_rate": learning_rate ,
                                            "cell_type": 'GRU', #'GRU', LSTM
                                            "opti": "Adam", #"SDG",
                                            "dataset": "eng-spa",
                                            "hidden_size": hidden_size,
                                            "batch_size": batch_size} , name="Opti Adam", tags=["Optimizer Adam"])


train(train_dataloader, val_dataloader, encoder, decoder, epoch, learning_rate =learning_rate, print_every=1, plot_every=5)

