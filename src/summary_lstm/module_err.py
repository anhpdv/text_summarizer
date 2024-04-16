import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from gensim.models import Word2Vec

try:
    from config.config import STOPWORDS_LINK, TEXT_SUMMARY_DATA
    from src.summary_lstm.preprocess_text import preprocess_text as tam_pre_text
except ImportError:
    from helpers import add_path_init
    from preprocess_text import preprocess_text as tam_pre_text

    add_path_init()
    from config import STOPWORDS_LINK, TEXT_SUMMARY_DATA

warnings.filterwarnings("ignore")


def load_data(data_link):
    if os.path.exists(data_link):
        data_train_test = pd.read_csv(data_link, nrows=10)
        print("Exit data preprocess")
    else:
        print("Start data preprocess")
        # Load Vietnamese stopwords from the custom file
        stopwords_link = STOPWORDS_LINK
        with open(stopwords_link, "r", encoding="utf-8") as file:
            stop_words_vietnamese = set(file.read().splitlines())
        data_text_summary = pd.read_csv(TEXT_SUMMARY_DATA)
        data_train_test = pd.DataFrame()
        # preprocess_text(data_text_summary["summary"][0], stop_words_vietnamese)
        data_train_test["summary"] = data_text_summary["summary"].apply(
            lambda x: tam_pre_text(x, stop_words_vietnamese)
        )
        data_train_test["fulltext"] = data_text_summary["fulltext"].apply(
            lambda x: tam_pre_text(x, stop_words_vietnamese)
        )
        data_train_test.to_csv(data_link, encoding="utf-8")

    # print(data_train_test.info())
    # print(data_train_test.head())

    return data_train_test


def train_word2vec_model(text_data, vector_size=100, window=5, min_count=1, epochs=10):
    word2vec_model = Word2Vec(
        sentences=text_data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
    )
    word2vec_model.train(text_data, total_examples=len(text_data), epochs=epochs)
    return word2vec_model


def preprocess_text_with_word2vec(text, word2vec_model, max_len):
    sequences = []
    for sentence in text:
        embeddings = [
            word2vec_model.wv[word].reshape(1, -1) for word in sentence if word in word2vec_model.wv
        ]
        sequences.append(embeddings)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_len, padding="post", dtype="float32"
    )
    return padded_sequences


def preprocess_summary(text, max_summary_len, tokenizer):
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_summary_len, padding="post"
    )
    return padded_sequences


def build_lstm_model(input_dim, output_dim, input_length, max_summary_len):
    # Instantiate the model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim,
                input_length=input_length,
            ),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    max_summary_len, activation="softmax"
                )  # Adjust output dimension to 80
            ),
        ]
    )

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )
    return history


def main():
    # Constants for hyperparameters
    max_text_len = 2800
    max_summary_len = 80
    output_dim = 100  # Size of Word2Vec embeddings

    # Load data
    data_link = "./dataset/data_train_test.csv"
    data_train_test = load_data(data_link)

    # Train Word2Vec model
    text_data = data_train_test["fulltext"].apply(lambda x: x.split()).tolist()
    word2vec_model = train_word2vec_model(text_data)

    # Preprocess text data using Word2Vec embeddings
    x_tr = data_train_test["fulltext"].apply(lambda x: x.split()).tolist()
    x_train_text = preprocess_text_with_word2vec(x_tr, word2vec_model, max_text_len)

    # Preprocess summary data
    y_tr = data_train_test["summary"].apply(lambda x: x.split()).tolist()
    tokenizer_summary = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_summary.fit_on_texts(y_tr)
    y_train_summary = preprocess_summary(y_tr, max_summary_len, tokenizer_summary)

    # Split data into train and validation sets
    X_train_text, X_val_text, y_train_summary, y_val_summary = train_test_split(
        x_train_text, y_train_summary, test_size=0.2, random_state=0, shuffle=True
    )

    # Build LSTM model
    input_shape = (max_text_len, output_dim)
    input_dim = output_dim  # Dimensionality of word embeddings
    input_length = max_text_len  # Length of each input sequence
    model_text = build_lstm_model(input_dim, output_dim, input_length, max_summary_len)

    # Train the model
    epochs = 10
    batch_size = 64
    history_text = train_model(
        model_text,
        X_train_text,
        y_train_summary,
        X_val_text,
        y_val_summary,
        epochs,
        batch_size,
    )

    # Additional code for evaluation or saving the model
    # ...


if __name__ == "__main__":
    main()
