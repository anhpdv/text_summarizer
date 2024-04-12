import pandas as pd


def main(stop_words_vietnamese):
    # Load Vietnamese stopwords from the custom file
    stopwords_link = "./models/stopwords/stopwords-vi.txt"
    with open(stopwords_link, "r", encoding="utf-8") as file:
        stop_words_vietnamese = set(file.read().splitlines())
    """
    Xử lý data train - test
    """
    # Apply preprocessing to the 'text' column
    # Data OCR còn lỗi chính tả - Có thể dung mô hình xử lý lỗi chính tả trước
    data_text = pd.read_csv("./dataset/data_full.csv")
    data_text.drop_duplicates(subset=["Journal_ID"], inplace=True)
    data_text.dropna(axis=0, inplace=True)
    
    data_preprocess = pd.DataFrame()
    data_preprocess["summary"] = data_text["Summary"]
    data_preprocess["text"] = data_text["full_text"]
    data_preprocess.to_csv("./dataset/data_preprocess.csv")
    # data_preprocess.to_excel("./dataset/data_preprocess.xlsx")
    # Print information about the preprocessed data
    # print(data_preprocess.info())
    # print(data_preprocess.head())


if __name__ == "__main__":
    main()
