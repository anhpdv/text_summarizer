# Đọc danh sách từ dừng từ tệp văn bản
def read_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        stopwords = [line.strip() for line in file]
    return stopwords


# Sắp xếp từ dừng theo thứ tự bảng chữ cái
def sort_stopwords(stopwords):
    # Loại bỏ từ trùng
    unique_stopwords = list(set(stopwords))
    # Sắp xếp từ dừng theo thứ tự bảng chữ cái
    sorted_stopwords = sorted(unique_stopwords)
    return sorted_stopwords


# In danh sách từ dừng đã sắp xếp
def print_stopwords(stopwords):
    for word in stopwords:
        print(word)


# Lưu danh sách từ dừng đã sắp xếp vào một tệp mới
def save_sorted_stopwords(sorted_stopwords, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for word in sorted_stopwords:
            file.write(word + "\n")


# Đường dẫn tới tệp chứa danh sách từ dừng
file_path = "./models/stopwords/my_stopwords.txt"

# Đọc và sắp xếp từ dừng
stopwords = read_stopwords(file_path)
sorted_stopwords = sort_stopwords(stopwords)

# Đường dẫn đến tệp đầu ra để lưu danh sách từ dừng đã sắp xếp
output_file = "./models/stopwords/my_stopwords.txt"

# Lưu danh sách từ dừng đã sắp xếp vào tệp mới
save_sorted_stopwords(sorted_stopwords, output_file)

print("Danh sách từ dừng đã được lưu vào tệp '{}'.".format(output_file))
