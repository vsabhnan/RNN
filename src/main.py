import pandas as pd
import nltk
from preprocess import preprocess, split_train_test, text_to_seq, make_padded_dataset
nltk.download('punkt_tab')
import torch, random, numpy as np

# Read dataset
df = pd.read_csv("/Users/varshasabhnani/Downloads/IMDB Dataset.csv")
df.head()

# Extract reviews and sentiments
texts = df["review"]
labels = df["sentiment"]
labels = labels.str.lower().map({"positive": 1, "negative": 0})

# Apply preprocessing steps
texts = texts.apply(preprocess)

word_counts = texts.str.split().apply(len)
print(word_counts.describe())

# Split into 50/50 training and testing sets
X_train, X_test, y_train, y_test = split_train_test(texts, labels)


# Convert into sequences of tokens
X_train_seq, X_test_seq = text_to_seq(X_train, X_test)

# Padded sequences of length 25, 50, and 100
X_train_25, y_train_25 = make_padded_dataset(X_train_seq, y_train, 25)
X_test_25, y_test_25 = make_padded_dataset(X_test_seq, y_test, 25)

X_train_50, y_train_50 = make_padded_dataset(X_train_seq, y_train, 50)
X_test_50, y_test_50 = make_padded_dataset(X_test_seq, y_test, 50)

X_train_100, y_train_100 = make_padded_dataset(X_train_seq, y_train, 100)
X_test_100, y_test_100 = make_padded_dataset(X_test_seq, y_test, 100)

datasets = {
    "25": {"train": (X_train_25, y_train_25), "test": (X_test_25, y_test_25)},
    "50": {"train": (X_train_50, y_train_50), "test": (X_test_50, y_test_50)},
    "100": {"train": (X_train_100, y_train_100), "test": (X_test_100, y_test_100)},
}

from evaluate import run_iter
run_index = 0

# Baseline parameters
def set_baseline():
    activation = "tanh"
    grad = False
    seq_length = 50
    opt = "adam"
    arch = "LSTM"
    return arch, activation, opt, grad, seq_length

# Effect of change in architecture
arch, activation, opt, grad, seq_length = set_baseline()
for arch in ["LSTM", "RNN", "BiLSTM"]:
        run_index += 1
        run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)

# Effect of change in activation function
arch, activation, opt, grad, seq_length = set_baseline()
for activation in ["sigmoid", "relu"]:
        run_index += 1
        run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)

# Effect of optimizer
arch, activation, opt, grad, seq_length = set_baseline()
for opt in ["sgd", "rmsprop"]:
        run_index += 1
        run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)

# Effect of sequence length
arch, activation, opt, grad, seq_length = set_baseline()
for seq_length in [25,100]:
        run_index += 1
        run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)

# Effect of gradient clipping
arch, activation, opt, grad, seq_length = set_baseline()

grad = True
run_index += 1
run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)
 

# Incremental changes for increased performance
arch, activation, opt, grad, seq_length = set_baseline()
seq_length = 100
activation = "sigmoid"

run_index += 1
run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)

grad = True
run_index += 1
run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)


# Compare best model with RNN and Bidirectional LSTM
seq_length = 100
activation = "sigmoid"
opt = "adam"
grad = False
arch = "RNN"
run_index += 1
run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)

arch = "BiLSTM"
run_index += 1
run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)

# Additional experiments with RNN
arch, activation, opt, grad, seq_length = set_baseline()
arch = "RNN"

for seq_length in [25,50,100]:
    for activation in ["tanh", "relu", "sigmoid"]:
        run_index += 1
        run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)

# Additional experiments with RNN
arch, activation, opt, grad, seq_length = set_baseline()
arch = "BiLSTM"
seq_length = 100

for activation in ["tanh", "relu"]:
    for grad in [True, False]:
        run_index += 1
        run_iter(run_index, arch, activation, opt, grad, seq_length, datasets)
