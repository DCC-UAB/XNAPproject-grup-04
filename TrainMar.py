from data_process import *
from model import *
import time
import math
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
import random
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import json

from data_process import prepareData
from model import EncoderRNN, AttnDecoderRNN

# Descarga el recurso 'wordnet'
nltk.download('wordnet')
nltk.download('omw-1.4')

SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
MAX_LENGTH = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def indexesFromSentence(lang, sentence):
    '''convert a sentence to indexes in the corpus'''
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    ''' sentence to index list to torch tensor'''
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    ''' get tensors for input and target language sentences'''
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def validate(encoder, decoder, validation_pairs, selected_pairs, max_length=MAX_LENGTH, criterion=nn.NLLLoss()):
    encoder.eval()   # Establece el modo de evaluación para el encoder
    decoder.eval()   # Establece el modo de evaluación para el decoder
    
    total_loss = 0
    total_bleu = 0
    total_meteor = 0
    num_sentences = len(validation_pairs)
    
    # Guardar las traducciones de las frases seleccionadas
    selected_translations = []

    with torch.no_grad():
        for idx, pair in enumerate(validation_pairs):
            input_sentence = pair[0]
            target_sentence = pair[1]
            
            input_tensor = tensorFromSentence(input_lang, input_sentence)
            target_tensor = tensorFromSentence(output_lang, target_sentence)
            input_length = input_tensor.size()[0]
            target_length = target_tensor.size()[0]
            
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden
            
            decoded_words = []
            loss = 0

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                
                if di < target_length:
                    loss += criterion(decoder_output, target_tensor[di])
                
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
                
                decoder_input = topi.squeeze().detach()

            total_loss += loss.item() / target_length
            
            reference = [target_sentence.split()]
            hypothesis = decoded_words[:-1] if decoded_words[-1] == '<EOS>' else decoded_words
            total_bleu += sentence_bleu(reference, hypothesis)
            total_meteor += meteor_score(reference, hypothesis)  # Asegúrate de que 'hypothesis' es una lista de palabras

            # Guardar la traducción si es una de las frases seleccionadas
            if idx in selected_pairs:
                selected_translations.append((input_sentence, ' '.join(hypothesis)))
    
    avg_loss = total_loss / num_sentences
    avg_bleu = total_bleu / num_sentences
    avg_meteor = total_meteor / num_sentences
    
    return avg_loss, avg_bleu, avg_meteor, selected_translations




def trainIters(encoder, decoder, n_epochs, train_pairs, val_pairs, print_every=1000, plot_every=100, learning_rate=0.01):
    print('Starting Training Loop...')
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    selected_indices = [2,4,6,8,10]

    # Estructura para almacenar las traducciones
    translations_per_epoch = []


    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}")
        for iter in range(1, len(train_pairs) + 1):
            training_pair = tensorsFromPair(random.choice(train_pairs))
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss
            
        # Validation at the end of each epoch
        val_loss, val_bleu, val_meteor, selected_translations = validate(encoder, decoder, val_pairs, selected_indices, criterion=criterion)
        
        print(f'Validation Loss: {val_loss:.4f}, Validation BLEU: {val_bleu:.4f}, Validation METEOR: {val_meteor:.4f}')

        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        wandb.log({"Validation Loss": val_loss, "Validation BLEU": val_bleu, "Validation METEOR": val_meteor,"Training loss": print_loss_avg}, step = epoch)
        print('%s (%d %d%%) %.4f' % (timeSince(start, iter / len(train_pairs)), iter, iter / len(train_pairs) * 100, print_loss_avg))

        # Guardar las traducciones en el diccionario
        epoch_translations = {
            "Epoch": epoch,
            "Sentences": []
        }
        
        for idx, (input_sentence, translation) in zip(selected_indices, selected_translations):
            epoch_translations["Sentences"].append({
                "Idx": idx,
                "Original": input_sentence,
                "Translation": translation
            })

        translations_per_epoch.append(epoch_translations)

        # Guardar las traducciones en un archivo JSON
        with open('translations_per_epoch.json', 'w') as json_file:
            json.dump(translations_per_epoch, json_file, ensure_ascii=False, indent=4)

 
    save_model(encoder, decoder)

def save_model(e, d):
    torch.save({'encoder': e.state_dict(), 'decoder': d.state_dict()}, './trained_model/seq2seq.net')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="no of epochs to train", default=10)
    parser.add_argument("--lr", help="learning rate", default=0.001)
    args = parser.parse_args()

    global input_lang, output_lang, pairs
    input_lang, output_lang, pairs = prepareData('eng', 'spa')
    
    # Split the pairs into training and validation sets
    train_size = 15000
    val_size = 5000
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    dropout = 0.1
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=dropout).to(device)

    wandb.init(project="Machine Translation", config={
        										"epochs": args.epochs, 
                                                "learning_rate": args.lr, 
                                                "cell_type": 'GRU', #'GRU', LSTM
                                                "opti": "SDG",
                                                "layers": 1,
                                                "dataset": "eng-spa",
                                                "hidden_size": hidden_size,
                                                "dropout": 0.1,
                                                "batch_size":1}, name="experiment1")

    trainIters(encoder1, attn_decoder1, int(args.epochs), train_pairs, val_pairs, print_every=5000, learning_rate=float(args.lr))

if __name__ == '__main__':
    main()
