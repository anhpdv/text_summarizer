import os
import pandas as pd

try:
    from src.summary_model.preprocess_text import preprocess_text
except ImportError:
    try:
        from summary_model.preprocess_text import preprocess_text
    except ImportError:
        from helpers import add_path_init

        add_path_init()
        from summary_model.preprocess_text import preprocess_text



def load_data(data_link):
    if os.path.exists(data_link):
        data_train_test = pd.read_csv(data_link)
        print("Exit data preprocess")
    else:
        print("Start data preprocess")
        data_text_summary = pd.read_csv("./dataset/data_text_summary.csv")
        data_train_test = pd.DataFrame()
        data_train_test["summary"] = data_text_summary["summary"].apply(
            lambda x: preprocess_text(x)
        )
        data_train_test["fulltext"] = data_text_summary["fulltext"].apply(
            lambda x: preprocess_text(x)
        )
        data_train_test.to_csv(data_link, encoding="utf-8")

    print(data_train_test.info())
    print(data_train_test.head())

    return data_train_test


def main():
    data_link = "./dataset/data_train_test.csv"
    load_data(data_link)


if __name__ == "__main__":
    main()
