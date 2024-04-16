import os
import sys
import re

from matplotlib import pyplot as plt
import pandas as pd
from underthesea import sent_tokenize, text_normalize, word_tokenize


def add_path_init():
    print("Add src to path.")
    current_directory = os.getcwd()
    directories = ["dataset", "config", "tools", "src"]
    for directory in directories:
        sys.path.insert(0, os.path.join(current_directory, directory))


def clean_text(text):
    # Common
    text = re.sub("(?s)<ref>.+?</ref>", "", text)  # remove reference links
    text = re.sub("(?s)<[^>]+>", "", text)  # remove html tags
    text = re.sub("&[a-z]+;", "", text)  # remove html entities
    text = re.sub("(?s){{.+?}}", "", text)  # remove markup tags
    text = re.sub("(?s){.+?}", "", text)  # remove markup tags
    text = re.sub("(?s)\[\[([^]]+\|)", "", text)  # remove link target strings
    text = re.sub("(?s)\[\[([^]]+\:.+?]])", "", text)  # remove media links
    text = re.sub(r"http[s]?://\S+", "", text) # Remove URLs
    text = re.sub(r"\b\d{10,}\b", "", text)
    text = re.sub("[']{5}", "", text)  # remove italic+bold symbols
    text = re.sub("[']{3}", "", text)  # remove bold symbols
    text = re.sub("[']{2}", "", text)  # remove italic symbols

    text = re.sub("[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)
    text = text.lower()

    # Common
    text = re.sub("[ ]{2,}", " ", text)  # Squeeze spaces.
    return text


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
        sentence_indices = [tokenizer.word_index.get(word, tokenizer.unk_index) for word in words]
        indices.append(sentence_indices)
    return indices


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
