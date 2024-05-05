from pyvi import ViTokenizer
# from load_data import data

def remove_extra_spaces(text):
    return ' '.join(text.split())


def replace_punctuation(text):
    text_with_periods = text.replace(';', '.').replace('!', '.').replace(':', '.').replace('@', '.').replace('_', ' ')
    return text_with_periods


import re
def keep_vietnamese_letters_and_numbers(text):
    return re.sub(r'[^\w\s.]', '', text)

def get_list_of_sentences(doc):
    sentences = []
    sens = doc.split('.')
    for sen in sens:
        if len(sen) > 35:
            # sen = gensim.utils.simple_preprocess(sen)
            # sen = ' '.join(sen)
            sen = ViTokenizer.tokenize(sen)
            sen = sen.lower()  # Chuyển đổi sang lowercase
            sen = sen.split(' ')
            sentences.append(sen)
    return sentences

def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = [word.strip() for word in file.readlines()]
    return stop_words

def remove_stop_words(text, stop_words):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def process_text(doc):
    doc = remove_extra_spaces(doc)
    doc = replace_punctuation(doc)
    # doc = remove_stop_words(doc,stopwords)
    doc = keep_vietnamese_letters_and_numbers(doc)
    sentences = get_list_of_sentences(doc)
    return sentences

def process_after(sentence_list):
    processed_text = []
    for sentence in sentence_list:
        processed_sentence = ' '.join([word for word in sentence]) + '.'
        processed_text.append(processed_sentence)
    return ' '.join(processed_text)
###############################
# stopwords = load_stop_words("./data/vietnamese-stopwords.txt")
# processed_text = [] 
# for entry in data:
#     text = entry['text']  
#     processed_entry = process_text(text) 
#     processed_text.append(processed_entry) 




# print(processed_text)
# word_counts = []  # Initialize an empty list to store word counts

# for entry in processed_text:
#     word_count = 0
#     for sentence in entry:
#         word_count += len(sentence)  # Đếm số từ trong mỗi đoạn văn
#     word_counts.append(word_count)  # Add word count for the current entry to the list

# print(word_counts)

# Iterate through the processed_text list
# for entry in processed_text:
#     for sentence in entry:
#         print(sentence)
#     print(entry)

