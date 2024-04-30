from PIL import Image
import os


def convert_images_to_png(folder_path):
    # Lấy danh sách các tệp trong thư mục
    files = os.listdir(folder_path)

    # Lọc ra các tệp ảnh
    image_files = [
        f
        for f in files
        if f.endswith(
            (".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".jfif", ".webp", ".png")
        )
    ]

    # Đảm bảo thư mục đích tồn tại
    output_folder = os.path.join(folder_path, "converted_images")
    os.makedirs(output_folder, exist_ok=True)

    # Đổi định dạng và đổi tên
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        output_path = os.path.join(output_folder, f"image{i+1}.png")

        # Mở ảnh
        with Image.open(image_path) as img:
            # Lưu dưới định dạng PNG
            img.save(output_path, "PNG")


def replate_class(folder_path):
    # file txt
    # Lấy danh sách các tệp trong thư mục
    files = os.listdir(folder_path)
    for txt_file in files:
        # Tạo đường dẫn đầy đủ đến tệp
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, "r") as file:
            lines = file.readlines()
        # Thay đổi giá trị đầu tiên của dòng đầu tiên thành "0"
        lines[0] = "0" + lines[0][1:]
        # Ghi dữ liệu mới vào file
        with open(file_path, "w") as file:
            file.writelines(lines)


if __name__ == "__main__":
    folder_path = "./datasets/data_mucluc/train/labels"
    # convert_images_to_png(folder_path)
    # replate_class(folder_path)
