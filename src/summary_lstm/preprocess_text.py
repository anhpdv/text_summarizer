from underthesea import sent_tokenize, word_tokenize, text_normalize


try:
    from src.summary_lstm.helpers import clean_text
except ImportError:
    from helpers import add_path_init, clean_text

    add_path_init()


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
