import re
from underthesea import sent_tokenize, word_tokenize, text_normalize

try:
    from config.config import STOPWORDS_USE
except ImportError:
    from helpers import add_path_init

    add_path_init()

    from config import STOPWORDS_USE

# Pre-compile regular expressions
HTML_TAG_RE = re.compile(r"<[^>]+>")
REF_LINK_RE = re.compile(r"<ref>.+?</ref>")
URL_RE = re.compile(r"http[s]?://\S+")
MARKUP_RE = re.compile(r"{{.+?}}|{.+?}")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Pre-compile regex pattern for save_key
SAVE_KEY_REGEX_PATTERN = re.compile(
    r"[^a-zA-Z0-9\s{}{}]".format(
        re.escape(
            "ỹáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ/"
        ),
        re.escape(
            "ỸÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ/"
        ),
    )
)


def clean_text(text, lower=False):
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
    text = EMAIL_RE.sub(".", text)  # Remove email addresses
    if lower:
        text = text.lower()
    text = SAVE_KEY_REGEX_PATTERN.sub("", text)

    # Squeeze spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text


def remove_stopwords(text, stop_words):
    # Chuyển đổi chuỗi văn bản thành danh sách các từ
    words = word_tokenize(text)
    # Loại bỏ các từ dừng
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Chuyển đổi danh sách các từ đã lọc thành chuỗi văn bản
    filtered_text = " ".join(filtered_words)
    return filtered_text


def preprocess_text(text, stop_words=STOPWORDS_USE, format_text=True):
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

    if format_text == True:
        # Join the cleaned sentences
        return_text = " ".join(cleaned_sentences)
        return_text = re.sub(r'\s([?.!",’])', r"\1", return_text)
        return return_text

    return cleaned_sentences


def main():
    text = "Sự phát triển của giáo dục đại học ngoài công lập là một xu thế tất yếu trong phát triển giáo dục trên thế giới. Giáo dục đại học ngoài công lập toàn cầu có quy mô sinh viên chiếm khoảng 1/3 tổng số sinh viên và đã có nhiều đóng góp quan trọng vào sự phát triển của giáo dục toàn cầu. Tuy nhiên, trong quá trình phát triển, bên cạnh những mặt mạnh và những yếu tố tích cực thì loại hình giáo dục đại học này ở từng khu vực, quốc gia trên thế giới cũng bộc lộ nhiều tồn tại, hạn chế và một số khuynh hướng phát triển tiêu cực. Bài viết này nghiên cứu và sử dụng những kết quả đánh giá chính của UNESCO năm 2021 về giáo dục đại học ngoài công lập toàn cầu, nghiên cứu thực trạng phát triển của giáo dục đại học tư thục Việt Nam, từ đó rút ra năm bài học kinh nghiệm đối với phát triển giáo dục đại học tư thục của Việt Nam. Trân Văn Hùng Phát triển giáo dục đại học ngoài công lập toàn câu - Bài học kinh nghiệm đối với Việt Nam Trân Văn Hùng Email: tranvanhung@duytan.edu.vn Trường Đại học Duy Tân Số 254 Nguyễn Văn Linh, thành phố Đà Nẵng, Việt Nam TÓM TẤT: Sự phát triển của giáo dục đại học ngoài công lập là một xu thế tất yếu trong phát triển giáo dục trên thế giới. Giáo dục đại học ngoài công lập toàn cầu có quy mô sinh viên chiếm khoảng 1/3 tổng số sinh viên và đã có nhiều đóng góp quan trọng vào sự phát triển của giáo dục toàn cầu. Tuy nhiên, trong quá trình phát triển, bên cạnh những mặt mạnh và những yếu tố tích cực thì loại hình giáo dục đại học này ở từng khu vực, quốc gia trên thế giới cũng bộc lộ nhiều tồn tại, hạn chế và một số khuynh hướng phát triển tiêu cực."
    text_word = clean_text(text)
    print(text_word)


if __name__ == "__main__":
    main()
