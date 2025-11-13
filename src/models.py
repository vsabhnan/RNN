import random, numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def build_model(architecture="LSTM",
                activation="tanh",
                optimizer="adam",
                clip=False,
                vocab_size=10000,
                embedding_dim=100,
                hidden_size=64,
                dropout_rate=0.4,
                sequence_length=25):

    # Set seed
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # First layer
    if architecture == "RNN":
        rnn_layer = SimpleRNN(hidden_size, activation=activation, return_sequences=True)
    elif architecture == "LSTM":
        rnn_layer = LSTM(hidden_size, activation=activation, return_sequences=True)
    elif architecture == "BiLSTM":
        rnn_layer = Bidirectional(LSTM(hidden_size, activation=activation, return_sequences=True))
    else:
        raise ValueError("architecture must be RNN, LSTM, or BiLSTM")

    # Second hidden layer (same architecture type but final output)
    if architecture == "RNN":
        rnn_layer_2 = SimpleRNN(hidden_size, activation=activation)
    elif architecture == "LSTM":
        rnn_layer_2 = LSTM(hidden_size, activation=activation)
    else:
        rnn_layer_2 = Bidirectional(LSTM(hidden_size, activation=activation))

    # Optimizer selection + optional gradient clipping
    if optimizer.lower() == "adam":
        opt = Adam(clipnorm=1.0) if clip else Adam()
    elif optimizer.lower() == "sgd":
        opt = SGD(clipnorm=1.0) if clip else SGD()
    elif optimizer.lower() == "rmsprop":
        opt = RMSprop(clipnorm=1.0) if clip else RMSprop()
    else:
        raise ValueError("Optimizer must be one of: adam, sgd, rmsprop")

    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        rnn_layer,
        Dropout(dropout_rate),
        rnn_layer_2,
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    return model
