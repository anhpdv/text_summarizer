import streamlit as st
import time
from preprocessing import process_text, process_after, replace_punctuation
from text_rank import textrank
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def main():
    st.title("TÓM TẮT VĂN BẢN")

    # Đầu vào cho văn bản 1
    text_input_1 = st.text_area("Nhập văn bản: ", value="", height=200)

    # Đầu vào cho văn bản 2
    text_input_2 = st.text_area("Văn bản đối sánh", value="", height=200)
    num_sen = st.number_input("Số câu", min_value=1, max_value=1000, value=3)

    # Nút nhấn để bắt đầu tóm tắt
    if st.button("Tóm tắt"):
        # Xử lý và tóm tắt văn bản 1
        start_time = time.time()
        summary_1 = process_after(textrank(process_text(text_input_1), num_sen))
        end_time = time.time()
        st.write("Thời gian :", end_time - start_time, " giây")
        st.write("Tóm tắt:", replace_punctuation(summary_1))

        # Xử lý và tóm tắt văn bản 2 nếu có
        if text_input_2:
            # summary_2 = []
            # for _ in process_text(text_input_2):
            #     summary_2.append(process_after(textrank(process_text())))
            # st.write("Tóm tắt văn bản 2:", summary_2)
            ori_text = process_after(process_text(text_input_2))
            # Tính điểm Rouge nếu cả hai ô nhập văn bản đều được điền
            scores = scorer.score(summary_1, ori_text)
            st.write("Rouge score:", scores)

if __name__ == "__main__":
    main()
