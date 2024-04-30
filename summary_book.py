import re
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
from config import config
from src.ocr.helpers import convert_contours_to_bounding_boxes
from src.ocr.ocr_image import detect_line_word, crop_box, detect_text_area
from src.ocr.pdf_to_img import pdf_to_image_np

import pytesseract


def load_model():
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_LINK
    return


def get_pdf_page_count(file_path):
    # Chuyển đổi tất cả các trang PDF thành hình ảnh
    images = convert_from_path(file_path)
    # Trả về số lượng hình ảnh, tương đương với số trang PDF
    return len(images)


def determine_summary_size(
    book_size: int, size_sumary: int, is_page=True, word_in_page=400
):
    """
    Determine the size of the summary in terms of number of words.

    Parameters:
        book_size (int): Number of pages in the book.
        size_sumary (int): Size of the summary in percentage (if is_page is False) or in pages (if is_page is True).
        is_page (bool): True if size_sumary is in pages, False if size_sumary is in percentage of book size.
        word_in_page (int): Average number of words per page.

    Returns:
        int: Size of the summary in terms of number of words.
    """
    if not is_page:
        w_summarize = book_size * size_sumary / 100 * word_in_page
    else:
        w_summarize = size_sumary * word_in_page
    return int(round(w_summarize))


def is_table_of_contents(text, chapter_lv1):
    # Chuyển đổi văn bản thành chữ thường
    text = text.lower()
    number_markers = ['1.', '2.', '3.', '4.',
                      '5.', '6.', '7.', '8.', '9.', '10.']

    # Tách các dòng văn bản thành từng dòng riêng biệt
    lines = text.split('\n')

    # Kiểm tra xem có ít nhất một dòng bắt đầu bằng số hoặc từ khóa
    for line in lines:
        if line.strip().startswith(chapter_lv1):
            return True

    # Kiểm tra xem có ít nhất một dòng chứa các tiêu đề có độ sâu lồng nhau không
    for i in range(len(lines)):
        if lines[i].strip().startswith(tuple(number_markers)):
            for j in range(i+1, min(i+4, len(lines))):
                if lines[j].strip().startswith(tuple(number_markers)):
                    return True
    return False


def extract_table_of_contents(list_page):
    # Các từ khóa tiêu đề và số
    title_keywords = ['chương', 'phần', "chapter"]
    number_chapter = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.',
                      '10.', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
    # Kết hợp danh sách title_keywords và number_chapter
    combined_markers = tuple(number_chapter + ['{} {}'.format(
        keyword, num) for keyword in title_keywords for num in range(1, 11)])

    page_menu = []
    for page in list_page:
        lines_rects = detect_line_word(page)
        list_test = []
        is_menu = 0
        for line in lines_rects:
            roi = crop_box(page, line)
            # text_line = detect_text_area(roi)
            text_line = pytesseract.image_to_string(roi, lang="vie")
            if len(text_line) != 0:
                text_line = re.sub(r"[\n]", " ", text_line)
                if is_table_of_contents(text_line, combined_markers):
                    is_menu += 1
                list_test.append(text_line)
        if is_menu > 4:
            page_menu.append(list_test)
        #     # Vẽ hình chữ nhật lớn xung quanh tất cả các hình chữ nhật con
        #     cv2.rectangle(page, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

        # plt.rcParams['figure.figsize'] = (16, 16)
        # plt.imshow(page)
        # plt.show()
    # print("page_menu", page_menu)
    return page_menu


def ocr_image(image):
    text = ""
    return text


# Vietnamese Ocr Correction - Tham khảo: https://github.com/buiquangmanhhp1999/VietnameseOcrCorrection
def correct_text(text):
    return text


# Trích xuất keyword văn bản
def keyword_text(text):
    key_word = []
    return key_word


# Tìm kiếm menu của sách
def find_menu(text):
    menu_book = []
    return menu_book


# Chia text theo chương
def divide_chapters(book):
    chapter_array = []
    return chapter_array


def sumary_text(text):
    text_sumary = ""
    return text_sumary


def sumary_book(book_link, size_sumary, is_page=True):
    load_model()

    word_in_page = 400
    text_sumary = "Nội dung tóm tắt"

    print("Step 1")
    # start = time.time()
    pages_np = pdf_to_image_np(book_link)
    len_pages = len(pages_np)
    words_to_summarize = determine_summary_size(
        len_pages, size_sumary, is_page)
    # end = time.time()
    print("Step 2")
    numbers_page_use_check_menu = round(len_pages * 0.2)
    print("numbers_page_use_check_menu", numbers_page_use_check_menu)
    start = time.time()
    list_page_check_menu = pages_np[:numbers_page_use_check_menu]
    # check_menu =
    extract_table_of_contents(list_page_check_menu)
    end = time.time()

    print(f"time Step 2:{(end-start):.03f}s")
    print(f"words_to_summarize:{words_to_summarize}")

    return text_sumary


def main():
    # book_link = "./datasets/data_book/Bup-Sen-xanh.pdf"
    book_link = "./datasets/data_book/TTX-Khung CPDT_V01_20180910.pdf"
    size_sumary = 20
    # Tóm tắt page - True | format % - False
    is_page = False
    sumary_book(book_link, size_sumary, is_page)
    # Input
    # Format: PDF
    # Số lượng trang: 51
    # --------------------------------
    # Bước 1: Xác định kích thước tóm tắt và tính toán số từ cần tóm tắt
    # Thời gian thực hiện: ~ 9.8s - 10.6s
    # Lượng từ tóm tắt: 4080 Từ
    # --------------------------------
    # Bước 2: Tìm kiếm mục lục sách và thực hiện OCR
    # 2.1. Tìm kiếm mục lục sách - Lấy 20% số trang tính từ bắt đầu thực hiện tìm kiếm
    return


if __name__ == "__main__":
    main()
