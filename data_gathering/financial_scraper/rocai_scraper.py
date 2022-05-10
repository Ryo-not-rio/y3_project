"""
Scrape financials from roic.ai by downloading the excel financials for a given company and converting the data into a csv.
"""

import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import sys
sys.path.insert(1, 'D:\\PycharmProjects\\y3_project')

root_dir = "D:\\PycharmProjects\\y3_project"
hdr = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'}

def get_excel(ticker):
    url = f"https://roic.ai/company/{ticker}"
    r = requests.get(url, headers=hdr, cookies={"twt": "ryo_not_rio"})
    soup = BeautifulSoup(r.text, features="lxml")
    links = soup.findAll('a')
    excel_link = ""
    for link in links:
        if link['href'][-5:] == ".xlsx":
            excel_link = link['href']

    if not excel_link:
        raise Exception("Excel link was not found")

    r = requests.get(excel_link, headers=hdr)
    with open(os.path.join(root_dir, "data", "tmp.xlsx"), "wb") as f:
        f.write(r.content)

def excel_to_data(file=os.path.join(root_dir, "data", "tmp.xlsx"), bdate="2018-01-01", adate="1999-12-31"):
    pd.set_option('display.max_columns', None)
    try:
        csv_df = pd.read_csv(os.path.join(root_dir, "data", "financials.csv"))
    except pd.errors.EmptyDataError:
        csv_df = pd.DataFrame()

    df = pd.read_excel(file, index_col=0, engine="openpyxl")
    df = df.rename(lambda x: str(x).strip(), axis=0)
    df = df.drop("SEC Link")
    columns = [x for x in df.columns if adate < str(x) < bdate]
    df = df.replace("- -", None)
    df = df[columns].dropna(axis=0, how="all").transpose()

    ticker = df.columns.name.split(" | ")[1]
    df.columns.name = ""
    df['year'] = df.index
    df.reset_index(level=0, inplace=True)
    df = df[['year'] + list(df.columns[:-1])]
    df.insert(loc=0, column="ticker", value=ticker)
    df = df.loc[:, ~df.columns.duplicated()] # Remove duplicate columns

    csv_df = csv_df.append(df).drop_duplicates()

    csv_df.to_csv(os.path.join(root_dir, "data", "financials.csv"), index=False)


def get_financials(ticker):
    print("Downloading financial excel sheet...")
    get_excel(ticker)
    print("Converting excel to csv...")
    excel_to_data()


if __name__ == "__main__":
    excel_to_data()