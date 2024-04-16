import pandas as pd
from transformers import pipeline

# Tạo một pipeline sử dụng mô hình tóm tắt
summarizer = pipeline("summarization")
data_link = "./dataset/data_train_test.csv"
df = pd.read_csv(data_link)

# Văn bản cần tóm tắt
text = df["fulltext"][0]

# Tóm tắt văn bản
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)

# In kết quả tóm tắt
print(summary[0]['summary_text'])
# Để tóm tắt văn bản bằng mô hình học máy, bạn có thể sử dụng một số phương hướng code sau:

# Sử dụng mô hình Transformer: Bạn có thể sử dụng một mô hình Transformer như BERT, GPT-3, hoặc T5 để tạo ra một mã nguồn tóm tắt văn bản. Các thư viện như Hugging Face's transformers cung cấp các công cụ mạnh mẽ để triển khai các mô hình này.

# Sử dụng mô hình seq2seq: Mô hình seq2seq với kiến trúc encoder-decoder cũng có thể được sử dụng để tạo ra mã nguồn tóm tắt văn bản. Bạn có thể sử dụng TensorFlow hoặc PyTorch để triển khai mô hình này.

# Sử dụng thư viện xử lý ngôn ngữ tự nhiên: Các thư viện như NLTK, spaCy, hoặc Gensim cung cấp các công cụ để xử lý văn bản và trích xuất thông tin quan trọng từ văn bản.