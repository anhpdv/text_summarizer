import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from underthesea import sent_tokenize, word_tokenize, text_normalize
import tensorflow as tf
from gensim.models import KeyedVectors 
from pyvi import ViTokenizer

try:
    from config.config import STOPWORDS_LINK, TEXT_SUMMARY_DATA
    from src.summary_lstm.helpers import check_text_word, check_rate_word
    from src.summary_lstm.preprocess_text import clean_text
    from summary_lstm.attention import AttentionLayer
    
except ImportError:
    from helpers import add_path_init, check_text_word, check_rate_word

    add_path_init()
    from config import STOPWORDS_LINK, TEXT_SUMMARY_DATA
    from preprocess_text import clean_text
    from summary_lstm.attention import AttentionLayer

# print(tf.config.list_physical_devices())
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


def remove_stopwords(text, vietnamese):
    # Chuyển đổi chuỗi văn bản thành danh sách các từ
    words = text.lower().split()
    # Loại bỏ các từ dừng
    filtered_words = [word for word in words if word not in vietnamese]
    # Chuyển đổi danh sách các từ đã lọc thành chuỗi văn bản
    filtered_text = " ".join(filtered_words)
    print(filtered_text)
    return filtered_text


def text_to_tokenize(text):
    # Normalize the text
    normalized_text = text_normalize(text)
    # Remove links, emails, and phone numbers
    cleaned_text = clean_text(normalized_text)
    # Tokenize into sentences
    sentences = sent_tokenize(cleaned_text)
    # Tokenize each sentence into words, remove stopwords and special characters, and lowercase
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        cleaned_sentences.append(words)
    return cleaned_sentences


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
    cleaned_text = clean_text(normalized_text)
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
        data_text_summary = pd.read_csv(TEXT_SUMMARY_DATA)
        data_train_test = pd.DataFrame()
        # preprocess_text(data_text_summary["summary"][0], stop_words_vietnamese)
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


def main():
    # Bước 1: Tiền xử lý dữ liệu
    data_link = "./dataset/data_train_test.csv"
    data_train_test = load_data(data_link)
    # data_text_summary = pd.read_csv(TEXT_SUMMARY_DATA)
    # text_x = text_to_tokenize(data_text_summary["summary"][0])
    # print(text_x)
    # Kiểm tra độ dài và các từ trong câu
    check_text_word(
        data_train_test,
        "summary",
        "fulltext",
        "./dataset/my_stopwords_data_preprocess_info.png",
        "./dataset/my_stopwords_group_preprocess_stats.csv",
    )

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


if __name__ == "__main__":
    main()
