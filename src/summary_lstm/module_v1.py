import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf

try:
    from config.config import STOPWORDS_LINK, TEXT_SUMMARY_DATA
    from src.summary_lstm.preprocess_text import preprocess_text as tam_pre_text
except ImportError:
    from helpers import add_path_init
    from preprocess_text import preprocess_text as tam_pre_text

    add_path_init()
    from config import STOPWORDS_LINK, TEXT_SUMMARY_DATA

# print(tf.config.list_physical_devices())
pd.set_option("display.max_colwidth", 200)
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


# Tiền xử lý dữ liệu văn bản và tóm tắt
def preprocess_text(text, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_len, padding="post"
    )
    return padded_sequences, tokenizer


def preprocess_summary(text, max_summary_len, tokenizer):
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_summary_len, padding="post"
    )
    # Convert sequences to one-hot encoding
    num_tokens = len(tokenizer.word_index) + 1
    one_hot_sequences = np.zeros(
        (len(padded_sequences), max_summary_len, num_tokens)
    )
    for i, sequence in enumerate(padded_sequences):
        for j, token_index in enumerate(sequence):
            one_hot_sequences[i, j, token_index] = 1
    return one_hot_sequences



# Xây dựng mô hình LSTM
def build_lstm_model(input_dim, output_dim, input_length, max_summary_len):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=input_dim, output_dim=output_dim, input_length=input_length
            ),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(max_summary_len, activation="softmax")
            ),
        ]
    )
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


# Huấn luyện mô hình
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, num_classes):
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    history = model.fit(
        X_train,
        y_train_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_one_hot),
    )
    return history


def main():
    # Bước 1: Tiền xử lý dữ liệu
    data_link = "./dataset/data_train_test.csv"
    data_train_test = load_data(data_link)
    # Độ dài tóm tắt chọn <= 80 từ| Văn bản <= 3000 Từ
    max_summary_len = 80
    max_text_len = 2800
    cleaned_text = np.array(data_train_test["fulltext"])
    cleaned_summary = np.array(data_train_test["summary"])

    short_text = []
    short_summary = []
    for i in range(len(cleaned_text)):
        if (
            len(cleaned_summary[i].split()) <= max_summary_len
            and len(cleaned_text[i].split()) <= max_text_len
        ):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])
    df = pd.DataFrame({"text": short_text, "summary": short_summary})
    # Thêm **START** và **END** tokens vào 2 đầu của summary (**start** - start of summary token, **end** - end of summary token)
    df["summary"] = df["summary"].apply(lambda x: "start " + x + " end")
    # Chia tập dữ liệu
    x_tr, x_val, y_tr, y_val = train_test_split(
        np.array(df["text"]),
        np.array(df["summary"]),
        test_size=0.2,
        random_state=0,
        shuffle=True,
    )

    x_train_text, tokenizer_text = preprocess_text(x_tr, max_text_len)
    x_train_summary, tokenizer_summary = preprocess_text(y_tr, max_summary_len)

    X_val_text, _ = preprocess_text(x_val, max_text_len)
    X_val_summary, _ = preprocess_text(y_val, max_summary_len)

    # Xây dựng mô hình LSTM
    input_dim_text = len(tokenizer_text.word_index) + 1
    input_dim_summary = len(tokenizer_summary.word_index) + 1
    output_dim = 128
    model_text = build_lstm_model(
        input_dim_text, output_dim, max_text_len, max_summary_len
    )

    # Huấn luyện mô hình
    epochs = 10
    batch_size = 64

    x_train_summary = preprocess_summary(y_tr, max_text_len, tokenizer_summary)
    X_val_summary = preprocess_summary(y_val, max_text_len, tokenizer_summary)
    num_classes = input_dim_text
    
    history_text = train_model(
        model_text,
        x_train_text,
        x_train_summary,
        X_val_text,
        X_val_summary,
        epochs,
        batch_size,
        input_dim_text,
    )


if __name__ == "__main__":
    main()
