
batch_size = 128  # Batch size for training.
epochs = 20  # Number of epochs to train for.
latent_dim = 1024 #256  # Latent dimensionality of the encoding space.
num_samples = 145437  # Number of samples to train on.

# Path to the data txt file on disk.
data_path = './Data/Cat-Eng/cat.txt' 
encoder_path='encoder_modelPredTranslation.h5'
decoder_path='decoder_modelPredTranslation.h5'

LOG_PATH="./log"

learingrate = 0.0001

opti = 'rmsprop' #'adam'

validation_split = 0.01


data_path = './Data/Cat-Eng/cat.txt' 
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

lines = open(data_path).read().split('\n')

print(lines)

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))