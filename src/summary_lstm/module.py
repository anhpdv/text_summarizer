import os
import pandas as pd
import warnings

try:
    from config.config import STOPWORDS_LINK, TEXT_SUMMARY_DATA
    from src.summary_lstm.preprocess_text import preprocess_text as tam_pre_text
    from src.summary_lstm.helpers import check_text_word
except ImportError:
    from helpers import add_path_init, check_text_word
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


def main():
    # Constants for hyperparameters
    max_text_len = 2800
    max_summary_len = 80
    output_dim = 100  # Size of Word2Vec embeddings

    # Load data
    data_link = "./dataset/data_train_test.csv"
    data_train_test = load_data(data_link)
    check_text_word(
        data_train_test,
        "summary",
        "fulltext",
        "./dataset/vietnamese_stopwords_data_preprocess_info.png",
        "./dataset/vietnamese_stopwords_group_preprocess_stats.csv",
    )


if __name__ == "__main__":
    main()
