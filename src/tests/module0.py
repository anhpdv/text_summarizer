import os
import numpy as np
import pandas as pd
import re
import warnings
from sklearn.model_selection import train_test_split
from underthesea import sent_tokenize, word_tokenize, text_normalize
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.layers import Reshape
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Embedding,
    Dense,
    Concatenate,
    TimeDistributed,
)
from keras_self_attention import SeqSelfAttention as AttentionLayer

try:
    from config.config import STOPWORDS_LINK
except ImportError:
    from helpers import add_path_init

    add_path_init()
    from config import STOPWORDS_LINK


# print(tf.config.list_physical_devices())
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


def remove_links_emails_phone(text):
    # Remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove phone numbers
    text = re.sub(r"\b\d{10,}\b", "", text)
    return text


def remove_stopwords(text, vietnamese):
    # Chuyển đổi chuỗi văn bản thành danh sách các từ
    words = text.lower().split()
    # Loại bỏ các từ dừng
    filtered_words = [word for word in words if word not in vietnamese]
    # Chuyển đổi danh sách các từ đã lọc thành chuỗi văn bản
    filtered_text = " ".join(filtered_words)
    print(filtered_text)
    return filtered_text


def preprocess_text(text, stop_words):
    """
    Preprocesses the input text by normalizing it, tokenizing it into sentences,
    and then tokenizing each sentence into words. It removes stopwords, special
    characters, and converts words to lowercase.

    Parameters:
        text (str): The input text to be preprocessed.
        stop_words (set): A set of stopwords to be removed from the text.

    Returns:
        str: The preprocessed text.

    Notes:
        - This function utilizes Underthesea's word_tokenize and sent_tokenize functions.
        - It tokenizes the input text into sentences and then tokenizes each sentence into words.
        - It removes stopwords, special characters, and converts words to lowercase.
    """
    # Normalize the text
    normalized_text = text_normalize(text)
    # Remove links, emails, and phone numbers
    cleaned_text = remove_links_emails_phone(normalized_text)
    # Tokenize into sentences
    sentences = sent_tokenize(cleaned_text)

    # Tokenize each sentence into words, remove stopwords and special characters, and lowercase
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [
            word.lower()
            for word in words
            if word.isalnum() and word.lower() not in stop_words
        ]
        if len(words) > 0:
            cleaned_sentence = " ".join(words)
            cleaned_sentences.append(cleaned_sentence)

    # Join the cleaned sentences
    cleaned_text = " ".join(cleaned_sentences)

    return cleaned_text


def load_data(data_link):
    if os.path.exists(data_link):
        data_train_test = pd.read_csv(data_link)
        print("Exit data preprocess")
    else:
        print("Start data preprocess")
        # Load Vietnamese stopwords from the custom file
        stopwords_link = STOPWORDS_LINK
        with open(stopwords_link, "r", encoding="utf-8") as file:
            stop_words_vietnamese = set(file.read().splitlines())
        data_text_summary = pd.read_csv("./dataset/data_text_summary.csv")
        data_train_test = pd.DataFrame()
        data_train_test["summary"] = data_text_summary["summary"].apply(
            lambda x: preprocess_text(x, stop_words_vietnamese)
        )
        data_train_test["fulltext"] = data_text_summary["fulltext"].apply(
            lambda x: preprocess_text(x, stop_words_vietnamese)
        )
        data_train_test.to_csv(data_link, encoding="utf-8")

    # print(data_train_test.info())
    # print(data_train_test.head())

    return data_train_test


def plot_word_count_distribution(length_df, column_names, colors):
    fig, axes = plt.subplots(len(column_names), 1, figsize=(10, 8))
    for i, (column_name, color) in enumerate(zip(column_names, colors)):
        length_df[column_name].plot.hist(bins=30, ax=axes[i], color=color, alpha=0.7)
        axes[i].set_title(f"{column_name} Word Count Distribution")
        axes[i].set_xlabel("Word Count")
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()


def calculate_group_stats(length_df, column_name, bins):
    length_df[f"{column_name}_group"] = pd.cut(length_df[column_name], bins=bins)
    group_stats = (
        length_df.groupby(f"{column_name}_group")
        .agg(
            count=pd.NamedAgg(column_name, "size"),
            average_word_count=pd.NamedAgg(column_name, "mean"),
        )
        .reset_index()
    )
    print(f"{column_name} Group Statistics:")
    print(group_stats)
    return group_stats


def check_text_word(
    data_df, column_name_1, column_name_2, save_path=None, save_statistics_path=None
):
    length_df = pd.DataFrame(
        {
            column_name_1: data_df[column_name_1].apply(lambda x: len(x.split())),
            column_name_2: data_df[column_name_2].apply(lambda x: len(x.split())),
        }
    )

    plot_word_count_distribution(
        length_df, [column_name_1, column_name_2], ["skyblue", "salmon"]
    )

    if save_path:
        plt.savefig(save_path)
        print(f"Plots saved to: {save_path}")
    else:
        plt.show()
    plt.close()

    bins_summary = [0, 10, 15, 20, 30, 40, 50, 100, 200]
    bins_text = [0, 250, 500, 750, 1000, 1500, 2000, 3500, 5000]
    # Data chưa tiền xử lý dùng
    # bins_summary = [0, 50, 100, 150, 200, 250, 300, 350]
    # bins_text = [0, 250, 500, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000]

    summary_group_stats = calculate_group_stats(length_df, column_name_1, bins_summary)
    text_group_stats = calculate_group_stats(length_df, column_name_2, bins_text)

    if save_statistics_path:
        # Concatenate summary and text group stats
        combined_stats = pd.concat([summary_group_stats, text_group_stats], axis=1)
        combined_stats.columns = [
            f"{column_name_1}_group",
            f"{column_name_1}_count",
            f"{column_name_1}_avg",
            f"{column_name_2}_group",
            f"{column_name_2}_count",
            f"{column_name_2}_avg",
        ]

        combined_stats.to_csv(save_statistics_path, index=False)

        print(f"Statistics saved to: {save_statistics_path}")


def check_rate_word(x_tokenizer):
    threshold = 4  # (từ 4 trở xuống là từ hiếm)
    thresh = 4

    count_false_threash = 0
    count_words = 0
    freq = 0
    tot_freq = 0

    for key, value in x_tokenizer.word_counts.items():
        count_words = count_words + 1
        tot_freq = tot_freq + value
        if value < thresh:
            count_false_threash = count_false_threash + 1
            freq = freq + value

    print(
        f"% of rare words in vocabulary:{round((count_false_threash / count_words) * 100, 2)}%"
    )
    print(f"Total Coverage of rare words:{round((freq / tot_freq) * 100, 2)}%")
    return count_words, count_false_threash


def main():
    # Bước 1: Tiền xử lý dữ liệu
    data_link = "./dataset/data_train_test.csv"
    data_train_test = load_data(data_link)

    # Kiểm tra độ dài và các từ trong câu
    # check_text_word(
    #     data_train_test,
    #     "summary",
    #     "fulltext",
    #     "./dataset/my_stopwords_data_preprocess_info.png",
    #     "./dataset/my_stopwords_group_preprocess_stats.csv",
    # )
    #     summary_group  count  average_word_count
    # 0       (0, 10]      1            7.000
    # 1      (10, 15]      1           12.000
    # 2      (15, 20]     12           17.500
    # 3      (20, 30]     30           27.200
    # 4      (30, 40]     54           35.241
    # 5      (40, 50]     54           45.444
    # 6     (50, 100]     50           60.060
    # 7    (100, 200]      2          118.500
    #       fulltext_group  count  average_word_count
    # 0       (0, 250]      1          104.000
    # 1     (250, 500]      0              NaN
    # 2     (500, 750]      0              NaN
    # 3    (750, 1000]      1          869.000
    # 4   (1000, 1500]     51         1345.961
    # 5   (1500, 2000]     90         1737.733
    # 6   (2000, 3500]     60         2312.000
    # 7   (3500, 5000]      1         3663.000
    # Đọ dài tóm tắt chọn <= 80 từ| Văn bản <= 3000 Từ
    max_summary_len = 80
    max_text_len = 3000
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
    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_tr))
    count_words, count_false_threash = check_rate_word(x_tokenizer)
    num_words_active = count_words - count_false_threash
    # num_words_active = 3041
    # Bỏ qua các từ xuất hiện ít
    x_tokenizer = Tokenizer(num_words=num_words_active)
    x_tokenizer.fit_on_texts(list(x_tr))

    x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
    x_val_seq = x_tokenizer.texts_to_sequences(x_val)

    y_tokenizer = Tokenizer(num_words=num_words_active)
    y_tokenizer.fit_on_texts(list(y_tr))

    y_tr_seq = y_tokenizer.texts_to_sequences(y_tr)
    y_val_seq = y_tokenizer.texts_to_sequences(y_val)

    # Padding
    x_tr_seq = pad_sequences(x_tr_seq, maxlen=max_text_len, padding="post")
    x_val_seq = pad_sequences(x_val_seq, maxlen=max_text_len, padding="post")

    y_tr_seq = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding="post")
    y_val_seq = pad_sequences(y_val_seq, maxlen=max_summary_len, padding="post")
    print("num_words: ", y_tokenizer.num_words)

    # vocab size (+1 padding token)
    x_voc = x_tokenizer.num_words + 1
    # size of vocabulary
    y_voc = y_tokenizer.num_words + 1

    print(y_tokenizer.word_counts["start"], len(y_tr))

    ind = []
    for i in range(len(y_tr)):
        cnt = 0
        for j in y_tr[i]:
            if j != 0:
                cnt = cnt + 1
        if cnt == 2:
            ind.append(i)

    y_tr = np.delete(y_tr, ind, axis=0)
    x_tr = np.delete(x_tr, ind, axis=0)
    ind = []
    for i in range(len(y_val)):
        cnt = 0
        for j in y_val[i]:
            if j != 0:
                cnt = cnt + 1
        if cnt == 2:
            ind.append(i)

    y_val = np.delete(y_val, ind, axis=0)
    x_val = np.delete(x_val, ind, axis=0)

    K.clear_session()

    latent_dim = 300
    embedding_dim = 100

    # Encoder
    encoder_inputs = Input(shape=(max_text_len,))

    # embedding layer
    enc_emb = Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)

    # encoder lstm 1
    encoder_lstm1 = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=0.4,
        recurrent_dropout=0.4,
    )
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    # encoder lstm 2
    encoder_lstm2 = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=0.4,
        recurrent_dropout=0.4,
    )
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    # encoder lstm 3
    encoder_lstm3 = LSTM(
        latent_dim,
        return_state=True,
        return_sequences=True,
        dropout=0.4,
        recurrent_dropout=0.4,
    )
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    # Set up the decoder, using `encoder_states` as initial state.
    # decoder_inputs = Input(shape=(None,))
    decoder_inputs = Input(shape=(max_summary_len,))

    # Lớp embedding
    dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=0.4,
        recurrent_dropout=0.2,
    )
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
        dec_emb, initial_state=[state_h, state_c]
    )
    decoder_outputs = Reshape((-1, latent_dim))(decoder_outputs)

    print("Encoder Outputs Shape:", encoder_outputs.shape)
    print("Decoder Outputs Shape:", decoder_outputs.shape)

    # Attention layer
    attn_layer = AttentionLayer()
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    # Nối attention input và decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name="concat_layer")(
        [decoder_outputs, attn_out]
    )

    # Lớp dense:
    decoder_dense = TimeDistributed(Dense(y_voc, activation="softmax"))
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.summary()


if __name__ == "__main__":
    main()
