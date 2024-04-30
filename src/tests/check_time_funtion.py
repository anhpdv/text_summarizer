import timeit


def check_pdf_to_images():
    # Define setup
    setup_code = """
try:
    from config import config
    from src.ocr.pdf_to_img import pdf_to_images
except ImportError:
    try:
        import config
        from ocr.pdf_to_img import pdf_to_images
    except ImportError:
        from helpers import add_path_init

        add_path_init()
        import config
        from ocr.pdf_to_img import pdf_to_images

output_folder = "./datasets/data_image/"
pdf_path = "./datasets/data_book/TTX-Khung CPDT_V01_20180910.pdf"
"""

    print("Start check")
    # Call timeit with function reference
    time_taken = timeit.timeit(
        stmt="pdf_to_images(pdf_path, output_folder)", setup=setup_code, number=1
    )
    print(f"Time taken:{time_taken} giây")


def check_pdf_to_image_np():
    # Define setup
    setup_code = """
try:
    from config import config
    from src.ocr.pdf_to_img import pdf_to_image_np
except ImportError:
    try:
        import config
        from ocr.pdf_to_img import pdf_to_image_np
    except ImportError:
        from helpers import add_path_init

        add_path_init()
        import config
        from ocr.pdf_to_img import pdf_to_image_np

output_folder = "./datasets/data_image/"
pdf_path = "./datasets/data_book/TTX-Khung CPDT_V01_20180910.pdf"
"""
    print("Start check")
    # Call timeit with function reference
    time_taken = timeit.timeit(
        stmt="pdf_to_image_np(pdf_path)", setup=setup_code, number=1
    )
    print(f"Time taken:{time_taken} giây")


# Thời gian thực thi: Độ phức tạp thời gian thực thi của mỗi hàm,
# tức là thời gian nó mất để thực hiện công việc.
# Hàm nào mất nhiều thời gian hơn để chạy có thể được xem là phức tạp hơn.


def main():
    print("Chọn funtion kiểm thử.")
    # "./datasets/data_book/TTX-Khung CPDT_V01_20180910.pdf"
    print("1 - check pdf_to_images")
    print("2 - check pdf_to_image_np")
    print("3 - EE")
    check_funt = ""
    while check_funt != "0":
        check_funt = input("Nhập bài tập số hoặc nhập 0 để thoát: ")
        match check_funt:
            case "1":
                check_pdf_to_images()
                # pdf_to_images - Time taken: ~ 21-23s - input- pdf - 51 page
            case "2":
                check_pdf_to_image_np()
                # pdf_to_image_np - Time taken: ~ 12-14s - input- pdf - 51 page
            case "0":
                break
            case "3":
                print("EE")
            case _:
                print("Vui lòng chọn bài tập hoặc ấn 0 để thoát.")


if __name__ == "__main__":
    main()
