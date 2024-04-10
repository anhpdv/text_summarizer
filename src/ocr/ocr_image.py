import csv
import os
import re
import cv2
import pytesseract
import openpyxl
from concurrent.futures import ThreadPoolExecutor

try:
    from config import config
except ImportError:
    from helpers import add_path_init

    add_path_init()
    import config


def ocr_tesseract(img):
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_LINK
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="vie")
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    cleaned_text = " ".join(text.split()).strip()
    return cleaned_text


def process_image(img_link):
    img = cv2.imread(img_link)
    return ocr_tesseract(img), os.path.basename(img_link)


def save_to_csv(img_folder_link, output_path):
    link_csv = os.path.join(output_path, "journal_text.csv")
    with open(link_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["journal", "text"])
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                process_image,
                [
                    os.path.join(img_folder_link, filename)
                    for filename in os.listdir(img_folder_link)
                    if filename.endswith(".png") or filename.endswith(".jpg")
                ],
            )
            for text, filename in results:
                writer.writerow([filename, text])


def save_to_xlsx(img_folder_link, output_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["journal", "text"])
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            process_image,
            [
                os.path.join(img_folder_link, filename)
                for filename in os.listdir(img_folder_link)
                if filename.endswith(".png") or filename.endswith(".jpg")
            ],
        )
        for text, filename in results:
            ws.append([filename, text])
    wb.save(os.path.join(output_path, "journal_text.xlsx"))


def main():
    img_folder_link = "./dataset/data_image/journal_9892_1/"
    output_path = config.TEXT_DATA_LINK
    os.makedirs(output_path, exist_ok=True)
    save_to_csv(img_folder_link, output_path)
    save_to_xlsx(img_folder_link, output_path)


if __name__ == "__main__":
    main()
