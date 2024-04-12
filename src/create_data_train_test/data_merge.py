import pandas as pd


data_summary = pd.read_csv("./dataset/data_csv_xlsx/data_summary.csv")
data_summary.drop_duplicates(subset=["Journal_ID"], inplace=True)
data_summary.dropna(axis=0, inplace=True)
data_summary.drop(columns=["Summary_Link"], inplace=True)
data_summary.drop(columns=["Unnamed: 0"], inplace=True)
data_summary["full_text"] = ""

for journal_id in data_summary["Journal_ID"]:
    data_text = pd.read_csv(f"./dataset/data_text/{journal_id}.csv")
    data_text.drop_duplicates(subset=["text"], inplace=True)
    data_text.dropna(axis=0, inplace=True)
    full_text = " ".join(data_text["text"])
    data_summary.loc[data_summary['Journal_ID'] == journal_id, 'full_text'] = full_text

# Remove "Tóm tắt:" from the beginning of each summary
data_summary["Summary"] = data_summary["Summary"].str.replace('Tóm tắt:', '', regex=False)
# Remove leading and trailing whitespaces
data_summary["Summary"] = data_summary["Summary"].str.strip()

data_summary.to_csv('./dataset/data_full.csv', encoding="utf-8")
data_summary.to_excel('./dataset/data_full.xlsx')
