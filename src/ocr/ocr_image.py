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
    try:
        import config
    except:
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


def ocr_easyocr(img):
    import easyocr

    reader = easyocr.Reader(["vi"])
    reader.readtext(img)


def process_image(img_link):
    img = cv2.imread(img_link)
    return ocr_tesseract(img), os.path.basename(img_link)


def save_to_csv(img_folder_link, output_path):
    csv_name = os.path.splitext(os.path.basename(img_folder_link))[0]
    link_csv = os.path.join(output_path, f"{csv_name}.csv")
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


def detect_line_word(image):
    if len(image.shape) == 3:
        # Nếu ảnh không phải là ảnh grayscale, chuyển đổi sang ảnh grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Nếu ảnh đã là ảnh grayscale, sử dụng ảnh đó trực tiếp
        gray = image

    # Sử dụng MSER để phát hiện vùng chữ
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    # Lọc các vùng quá nhỏ
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    hulls = [h for h in hulls if cv2.contourArea(h) > 100]

    # Sắp xếp các vùng theo thứ tự từ trên xuống dưới
    hulls.sort(key=lambda x: cv2.boundingRect(x)[1])
    # Chia ảnh thành các dòng chữ
    lines = []
    line = []
    for hull in hulls:
        x, y, w, h = cv2.boundingRect(hull)
        if line and y - line[-1][1] > h * 1.2:  # Nếu khoảng cách lớn hơn 1.2 lần chiều cao
            lines.append(line)
            line = []
        line.append((x, y, x + w, y + h))
    if line:
        lines.append(line)
    # Tính toán khoảng cách trung bình giữa các dòng chữ
    avg_line_spacing = sum(line[0][1] - line[-1][3] for line in lines) / len(lines)

    # Tính toán padding
    padding = int(avg_line_spacing / 2)
    lines_text = []
    line_text = []
    for line in lines:
        # Tính toán tọa độ của hình chữ nhật lớn nhất chứa tất cả các hình chữ nhật con của các dòng chữ
        x_min = min(rect[0] for rect in line) + padding
        y_min = min(rect[1] for rect in line) + padding
        x_max = max(rect[2] for rect in line) - padding
        y_max = max(rect[3] for rect in line) - padding
        line_text.append((x_min, y_min, x_max, y_max))
    if line_text:
        lines_text.append(line_text)
    return lines_text

def main():
    output_path = config.TEXT_DATA_LINK
    os.makedirs(output_path, exist_ok=True)
    # img_folder_link = "./datasets/data_image/journal_9892_1/"
    # save_to_csv(img_folder_link, output_path)
    # save_to_xlsx(img_folder_link, output_path)
    pdf_folder_path = "./datasets/data_image/"
    # for filename in os.listdir(pdf_folder_path):
    #     img_folder_link = os.path.join(pdf_folder_path, filename)
    #     print(f"Convert image {img_folder_link}")
    #     save_to_csv(img_folder_link, output_path)
    link = r"D:\Product\text_summarizer\datasets\Screenshot 2024-04-24 003118.png"
    img = cv2.imread(link)
    # print(ocr_tesseract(img))
    # 11. 1.2. 1.3. 24. 22. 221. 222. 2.23. 224. 225. 2.3. 2.31. 2.32. 4. 4.2 43 54. 52. s43. s4. s5, s6. 57. 524. 522, 5.23. 54. s4. S1. s42, 5.3, 61. 6.2. 6.3. Z1 z2
    # Mục đích và phạm vi áp dụng kiến trúc Chính phủ điện tử TIXVN Giới thiệu chung về Khung kiến trúc Chính phủ điện tử: Sự cần thiết xây dựng Kiến trúc Chính phủ điện tử tại Thông tấn xã Việt Nam:.. Mục đích và phạm vi áp dụng.. Hiện trạng hệ thống Công nghệ thông tin và Chính phủ điện tử tại Thông tấn xã Hiện trạng triển khai ứng dụng CNTT tại Thông tấn xã Việt Nam: Các hệ thống ứng dụng CNTT của TTXVN. Hệ thống tác nghiệp phục vụ sản xuất thông tỉn.. Các ứng dụng của Hệ thống kỹ thuật sản xuất tin truyền hình.. Hệ thống phục vụ quản lý hành chính nhà nước. Trung tâm dữ liệu. Hệ thống đảm bảo an toàn thông tin... Hiện trạng triển khai CPĐT tại TTXVN.. Khung kiến trúc CPĐT Việt Nam... Đánh giá. Định hướng Kiến trúc Chính phủ điện tử Thông tấn xã Chức năng, nhiệm vụ của TTXVN. Cơ cấu tổ chức TTXVN. Tầm nhìn, định hướng phát triển Chính phủ điện tử TTXVN. Các nguyên tắc xây dựng kiến trúc Chính phủ điện tử TTXVN: Kiến trúc Chính phủ điện tử Thông tấn xã Sơ đồ tổng thể Kiến trúc Chính phủ điện tử TTXVN.. lệt Nam... Người sử dụng Kênh truy cập Dịch vụ cổng thông tin điện tử... Dịch vụ công trực tuyến (Dịch vụ thông tin)..... Ứng dụng và cơ sở dữ liệu Các dịch vụ chia sẻ và tích hợp.... Nền tảng tích hợp dịch vụ CPĐT (LGSP)... Nền tảng dịch vụ dùng chung..... Nền tảng tích hợp ứng dụng. Các dịch vụ tích hợp và liên thông dữ liệu... Các nguyên tắc, yêu cầu trong việc triển khai các thành phần trong Kiến trúc CPĐT TTXVN.... Nguyên tắc... Yêu cầu về nghiệp vụ Yêu cầu về kỹ thuật... Lộ trình/kế hoạch/nguồn kinh phí 1ộ trình triển khai kiến trúc Chính phủ Kế hoạch triển khai... Công tác chỉ đạo triển khai kiến trúc CPĐT TTXVN.... Công tác quản lý, giám sát, duy trì Kiến trúc CPĐT TTXVN.... =Ô -10 _ -12 -18 -18 -15 -15 -16 -17 -19 .21 .21 -2 -23 -23 -28 -28 -30 .31 .32 -34 -36 -37 -37 -38 -38 -42 -42 -48 -46 -46 -46 -46

if __name__ == "__main__":
    main()
