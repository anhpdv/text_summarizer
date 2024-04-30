from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import numpy as np
from pdf2image import convert_from_path

try:
    from config import config
except ImportError:
    try:
        import config
    except:
        from helpers import add_path_init

        add_path_init()
        import config


def pdf_to_images(pdf_path, output_folder):
    """
    Convert each page of a PDF file to an image and save them in the output folder.

    Parameters:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder where the converted images will be saved.

    Returns:
        None

    Notes:
        - This function converts each page of the PDF file to a PNG image.
        - It creates a subfolder in the output folder with the name of the PDF file (without extension).
        - It saves each image with the format "{pdf_name}_page_{page_number}.png" in the subfolder.
        - It uses pdf2image library to perform the conversion.
        - The function assumes that the poppler library path is specified in the config module.
    """
    try:
        # Check if the output folder exists, if not create it
        os.makedirs(output_folder, exist_ok=True)

        # Get the base name of the PDF file
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Convert{pdf_name}")
        pdf_to_img_fodler = os.path.join(output_folder, pdf_name)
        os.makedirs(pdf_to_img_fodler, exist_ok=True)
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, poppler_path=config.POPPLER_PATH)

        # Process each page
        for i, page in enumerate(pages):
            # Define the filename
            filename = f"{pdf_name}_page_{i}.png"
            # Define the filepath
            filepath = os.path.join(pdf_to_img_fodler, filename)

            # Save the page as a PNG image
            page.save(filepath, "PNG")

            # Optionally, if you want to use OpenCV to save the image
            # Convert the PIL image to a numpy array
            page_np = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

            # Save the numpy array as an image using OpenCV
            cv2.imwrite(filepath, page_np)

        print("Conversion completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")


def pdf_to_image_np(pdf_path):
    """
    Convert each page of a PDF file to a NumPy array representing an image.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: List of NumPy arrays representing each page of the PDF file.

    Notes:
        - This function converts each page of the PDF file to a NumPy array.
        - It uses pdf2image library to perform the conversion.
        - The function assumes that the poppler library path is specified in the config module.
    """
    try:
        print(f"Start Convert")
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, poppler_path=config.POPPLER_PATH)

        # List to store NumPy arrays of pages
        page_np_list = []

        # Process each page
        for page in pages:
            # Convert the PIL image to a numpy array
            page_np = np.array(page)
            # Append the numpy array to the list
            page_np_list.append(page_np)

        return page_np_list

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main():
    output_folder = config.IMAGE_DATA_LINK
    # Đường dẫn đến tài liệu PDF chứa bảng
    # pdf_path = "./datasets/data_pdf/journal_9892_1.pdf"
    # pdf_to_images(pdf_path, output_folder)
    pdf_folder_path = "./datasets/data_pdf"
    for filename in os.listdir(pdf_folder_path):
        pdf_path = os.path.join(pdf_folder_path, filename)
        # pdf_to_images(pdf_path, output_folder)


if __name__ == "__main__":
    main()
