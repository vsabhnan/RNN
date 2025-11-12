import re
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def preprocess(text):
    """
        Input: review text
        Output: tokenized review text
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # keep letters, digits, whitespace
    tokens = word_tokenize(text)
    return " ".join(tokens)

def split_train_test(text, labels):
    """
        Input: Tokenized reviews
        Output: Train and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        text, labels, test_size=0.5, random_state=42, shuffle=False
    )

    return X_train, X_test, y_train, y_test

def text_to_seq(X_train, X_test):
    """
        Input: Train and test reviews
        Output: Sequenced tokens based on top 10000 words
    """
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    total_vocab = len(tokenizer.word_index)
    print(f"Total unique words in training set: {total_vocab}")

    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    return X_train_seq, X_test_seq

def make_padded_dataset(sequences, labels, length):
    """
        Input: Sequenced tokens
        Output: Padded sequences of specified length
    """
    X = pad_sequences(sequences, maxlen=length, padding='post', truncating='post')
    return X, labels