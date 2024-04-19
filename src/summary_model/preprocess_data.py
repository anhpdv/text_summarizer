# Tiền xử lý dữ liệu mô hình
import os
import re
from matplotlib import pyplot as plt
import pandas as pd
from underthesea import sent_tokenize, word_tokenize, text_normalize

try:
    from config.config import STOPWORDS_USE, TEXT_SUMMARY_DATA
except ImportError:
    from helpers import add_path_init

    add_path_init()

    from config import STOPWORDS_USE, TEXT_SUMMARY_DATA


# Pre-compile regular expressions
HTML_TAG_RE = re.compile(r"<[^>]+>")
REF_LINK_RE = re.compile(r"<ref>.+?</ref>")
URL_RE = re.compile(r"http[s]?://\S+")
MARKUP_RE = re.compile(r"{{.+?}}|{.+?}")


def clean_text(text):
    # Convert text to string only once
    text = str(text)

    # Common substitutions
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"[-_~`]{2,}", "", text)
    text = re.sub(r"```+|\.\.\.+", "", text)
    text = re.sub(r"[']{2,5}", "", text)
    text = re.sub(r"[<>()|[&\[\]\"\",;?*!]", "", text)

    # Remove specific patterns using pre-compiled regular expressions
    text = REF_LINK_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MARKUP_RE.sub("", text)

    text = text.lower()
    # Loại bỏ các ký tự đặc biệt nhưng giữ lại các ký tự tiếng Việt và dấu cách
    save_key = "ỹáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ/"
    # Tạo biểu thức chính quy từ các ký tự trong save_key
    regex_pattern = r'[^a-zA-Z0-9\s{}]'.format(re.escape(save_key))
    text = re.sub(regex_pattern, '', text)

    # Squeeze spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text


def remove_stopwords(text, stop_words):
    # Chuyển đổi chuỗi văn bản thành danh sách các từ
    words = text.lower().split()
    # Loại bỏ các từ dừng
    filtered_words = [word for word in words if word not in stop_words]
    # Chuyển đổi danh sách các từ đã lọc thành chuỗi văn bản
    filtered_text = " ".join(filtered_words)
    # print(filtered_text)
    return filtered_text


def preprocess_text(text, stop_words=STOPWORDS_USE):
    # Normalize the text
    normalized_text = text_normalize(text)
    # Clear tag, link and lower word
    cleaned_text = clean_text(normalized_text)
    # Tokenize into sentences
    sentences = sent_tokenize(cleaned_text)
    # Tokenize each sentence into words, remove stopwords and special characters, and lowercase
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        # Remove stopwords and non-alphanumeric characters, and lowercase
        cleaned_words = [word for word in words if word.isalnum() not in stop_words]
        if cleaned_words:
            cleaned_sentence = " ".join(cleaned_words)
            cleaned_sentences.append(cleaned_sentence)

    # Join the cleaned sentences
    return_text = " ".join(cleaned_sentences)
    return_text = re.sub(r'\s([?.!",’])', r"\1", return_text)
    print(f"return_text:{return_text}")
    return return_text


def text_to_index(text, tokenizer):
    # Normalize the text
    normalized_text = text_normalize(text)
    # Remove links, emails, and phone numbers
    cleaned_text = clean_text(normalized_text)
    # Tokenize into sentences
    sentences = sent_tokenize(cleaned_text)
    # Convert sentences to indices
    indices = []
    for sentence in sentences:
        # Tokenize each sentence into words
        words = word_tokenize(sentence)
        # Encode words into indices
        sentence_indices = [
            tokenizer.word_index.get(word, tokenizer.unk_index) for word in words
        ]
        indices.append(sentence_indices)
    return indices


def check_rate_word(x_tokenizer):
    # (từ 4 trở xuống là từ hiếm)
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


def load_data(data_link):
    if os.path.exists(data_link):
        data_train_test = pd.read_csv(data_link)
        print("Exit data preprocess")
    else:
        print("Start data preprocess")
        data_text_summary = pd.read_csv(TEXT_SUMMARY_DATA)
        data_train_test = pd.DataFrame()
        data_train_test["summary"] = data_text_summary["summary"].apply(
            lambda x: preprocess_text(x, STOPWORDS_USE)
        )
        data_train_test["fulltext"] = data_text_summary["fulltext"].apply(
            lambda x: preprocess_text(x, STOPWORDS_USE)
        )
        data_train_test.to_csv(data_link, encoding="utf-8")

    return data_train_test


def main():
    # Bước 1: Tiền xử lý dữ liệu
    data_link = "./dataset/data_train_test.csv"
    data_train_test = load_data(data_link)


if __name__ == "__main__":
    # main()
    text = "Sự phát triển của giáo dục đại học ngoài công lập:}{:@$%:}}{}@$%@$% là một xu thế tất yếu trong phát triển giáo dục trên thế giới. Giáo dục đại học ngoài công lập toàn cầu có quy mô sinh viên chiếm khoảng 1/3 tổng số sinh viên và đã có nhiều đóng góp quan trọng vào sự phát triển của giáo dục toàn cầu. Tuy nhiên, trong quá trình phát triển, bên cạnh những mặt mạnh và những yếu tố tích cực thì loại hình giáo dục đại học này ở từng khu vực, quốc gia trên thế giới cũng bộc lộ nhiều tồn tại, hạn chế và một số khuynh hướng phát triển tiêu cực. Bài viết này nghiên cứu và sử dụng những kết quả đánh giá chính của UNESCO năm 2021 về giáo dục đại học ngoài công lập toàn cầu, nghiên cứu thực trạng phát triển của giáo dục đại học tư thục Việt Nam, từ đó rút ra năm bài học kinh nghiệm đối với phát triển giáo dục đại học tư thục của Việt Nam."
    preprocess_text(text, stop_words=STOPWORDS_USE)

# check: BLUE 1,2,3,4
# capa index
