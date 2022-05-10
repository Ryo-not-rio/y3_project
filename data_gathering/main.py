"""
Scrape the web for MD&A, financials and prices
"""

import os
import shutil
import pandas as pd
from datetime import datetime, timedelta

import text_scraper.TenK as TenK
from financial_scraper import rocai_scraper
from price_scraper import price_scraper

root_dir = "D:\\PycharmProjects\\y3_project"

def get_mda(ticker):
    downloader = TenK.TenKDownloader([ticker], "20030101", "20181231")
    scraper = TenK.TenKScraper('Item 7', 'Item 8')
    data_dir = os.path.join(root_dir, "data")
    tmp_dir = os.path.join(root_dir, "data", "tmp")

    try:
        os.mkdir(tmp_dir)
    except FileExistsError:
        pass

    downloader.download(tmp_dir)

    dates = []
    for file in os.listdir(tmp_dir):
        date = file.split("_")[1][:-4]
        date = date[:4] + "-" + date[4:6] + "-" + date[6:]
        dates.append(date)
        try:
            print(f"Parsing {file} MD&A...")
            scraper.scrape(os.path.join(tmp_dir, file), os.path.join(data_dir, "mda", f"{file[:-3]}txt"))
        except:
            continue

    shutil.rmtree(tmp_dir)
    return dates


if __name__ == "__main__":
    tickers = []
    with open("active.csv") as f:
        df = pd.read_csv(f)

    for i, row in df.iterrows():
        if row["status"] == "Delisted" and row["delistingDate"] < "2005-01-01":
            continue

        ticker = row["symbol"]
        try:
            dates = get_mda(ticker)  # Download mda
            rocai_scraper.get_financials(ticker)  # Download financials

            # Download prices
            for date in dates:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                search_dates = [date_obj + timedelta(days=365), date_obj + timedelta(days=365 * 3)]
                for d in search_dates:
                    date_str = datetime.strftime(d, "%Y-%m-%d")
                    print(f"Downloading price for {date_str}...")
                    try:
                        price_scraper.get_price(ticker, date_str)
                    except:
                        print(f"price for {date_str} not found")
                        continue

            with open(os.path.join(root_dir, "data", "prices.csv")) as f:
                price_df = pd.read_csv(f).drop_duplicates()
            with open(os.path.join(root_dir, "data", "prices.csv"), "w", newline="") as f:
                price_df.to_csv(f, index=False)

        except Exception as e:
            print(f"{ticker} skipped: Error - {e}")
        finally:
            df["scraped"][i] = True
            with open(os.path.join(root_dir, "data_gathering", "active.csv"), "w", newline="") as f:
                df.to_csv(f)
            print(f"Processed {ticker}")










