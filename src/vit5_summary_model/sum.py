import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Dowload model in https://huggingface.co/VietAI/vit5-large-vietnews-summarization

print("Load token")

tokenizer = AutoTokenizer.from_pretrained("./src/vit5_summary_model")
model = AutoModelForSeq2SeqLM.from_pretrained("./src/vit5_summary_model")
model.cuda()
# Load data
print("Load data")
data_link = "./dataset/data_text_summary.csv"
data_train_test = pd.read_csv(data_link, nrows=10)
summary = data_train_test["summary"][2]
print("summary:", summary)
sentence = data_train_test["fulltext"][2]
# sentence = "Sự phát triển của giáo dục đại học ngoài công lập là một xu thế tất yếu trong phát triển giáo dục trên thế giới. Giáo dục đại học ngoài công lập toàn cầu có quy mô sinh viên chiếm khoảng 1/3 tổng số sinh viên và đã có nhiều đóng góp quan trọng vào sự phát triển của giáo dục toàn cầu. Tuy nhiên, trong quá trình phát triển, bên cạnh những mặt mạnh và những yếu tố tích cực thì loại hình giáo dục đại học này ở từng khu vực, quốc gia trên thế giới cũng bộc lộ nhiều tồn tại, hạn chế và một số khuynh hướng phát triển tiêu cực. Bài viết này nghiên cứu và sử dụng những kết quả đánh giá chính của UNESCO năm 2021 về giáo dục đại học ngoài công lập toàn cầu, nghiên cứu thực trạng phát triển của giáo dục đại học tư thục Việt Nam, từ đó rút ra năm bài học kinh nghiệm đối với phát triển giáo dục đại học tư thục của Việt Nam."
text = "vietnews: " + sentence + " </s>"
encoding = tokenizer(text, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding[
    "attention_mask"
].to("cuda")
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_masks,
    max_length=256,
    early_stopping=True,
)
for output in outputs:
    line = tokenizer.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(line)
