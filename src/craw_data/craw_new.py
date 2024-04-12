import requests
from bs4 import BeautifulSoup
import csv
import os

os.makedirs("data_craw",exist_ok=True)

def get_summary_text(url):
    """
    Extracts and returns the summary text from the given URL.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    summary_element = soup.find("div", class_="field-name-field-news-mota")
    if summary_element:
        return summary_element.text.strip()
    else:
        return "Summary not found"

def main():
    number_start = 9890
    number_end = 10965
    url_parent = "http://vjes.vnies.edu.vn"

    # Open a CSV file for writing
    with open("summary.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Summary_Link", "Journal_ID", "Summary"])

        for journal_id in range(number_start, number_end):
            url = f"http://vjes.vnies.edu.vn/vi?year=2023&journal={journal_id}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            views_rows = soup.find_all(class_='views-row')

            if views_rows:
                print(f"Processing journal_id:{journal_id}")
                for i, row in enumerate(views_rows, 1):
                    summary_link = row.select_one('.column1-title a')
                    if summary_link:
                        b_summary_link = url_parent + summary_link['href']
                    else:
                        b_summary_link = None

                    pdf_link = row.select_one('.column1-btn a')
                    if pdf_link:
                        pdf_link = pdf_link['href']
                        # Download the PDF file
                        pdf_response = requests.get(pdf_link)
                        with open(f"data_craw/journal_{journal_id}_{i}.pdf", "wb") as f:
                            f.write(pdf_response.content)

                    # Extract summary text
                    if b_summary_link:
                        summary_text = get_summary_text(b_summary_link)
                    else:
                        summary_text = "Summary link not found"

                    # Write to CSV
                    writer.writerow([b_summary_link, f"journal_{journal_id}_{i}", summary_text])
                    print(f"Downloaded: journal_{journal_id}_{i}.pdf")

if __name__ == "__main__":
    main()
