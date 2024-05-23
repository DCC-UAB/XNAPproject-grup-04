
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import pandas as pd
import spacy
import random
from torchtext.data.metrics import bleu_score
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


spacy_spanish = spacy.load("es")
spacy_english = spacy.load("en")

def tokenize_spanish(text):
  return [token.text for token in spacy_spanish.tokenizer(text)]

def tokenize_english(text):
  return [token.text for token in spacy_english.tokenizer(text)]



spanish = Field(tokenize=tokenize_spanish,
               lower=True,
               init_token="<sos>",
               eos_token="<eos>")

english = Field(tokenize=tokenize_english,
               lower=True,
               init_token="<sos>",
               eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts = (".es", ".en"),
                                                    fields=(spanish, english))

spanish.build_vocab(train_data, max_size=10000, min_freq=3)
english.build_vocab(train_data, max_size=10000, min_freq=3)