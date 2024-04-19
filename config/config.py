# Paths and formats:
import torch


PRIMARY_DATA_LINK = "./dataset/data_pdf"
PDF_DATA_LINK = "./dataset/data"
IMAGE_DATA_LINK = "./dataset/data_image"
TEXT_DATA_LINK = "./dataset/data_text/"
POPPLER_PATH = "./tools/poppler-24.02.0/Library/bin"
SAVE_MODEL_PATH = "./models"
SAVE_LOG_PATH = "./log"
TESSERACT_LINK = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
STOPWORDS_LINK = "./models/stopwords/mini_stopwords.txt"
TEXT_SUMMARY_DATA = "./dataset/data_text_summary.csv"

try:
    with open(STOPWORDS_LINK, "r", encoding="utf-8") as file:
        STOPWORDS_USE = set(file.read().splitlines())
except FileNotFoundError:
    STOPWORDS_USE = set()  # Set to an empty set if the file doesn't exist
except Exception as e:
    print("An error occurred while reading stopwords:", e)
    STOPWORDS_USE = set()  # Set to an empty set if there's any other error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{DEVICE}")
