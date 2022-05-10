import yfinance as yf
from datetime import datetime, timedelta
import csv
import os
import requests_cache

root_dir = "D:\\PycharmProjects\\y3_project"

def get_price(ticker, date):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    end_date = datetime.strftime(date_obj + timedelta(days=10), "%Y-%m-%d")
    session = requests_cache.CachedSession('yfinance.cache')
    session.headers['User-agent'] = 'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
    downloader = yf.Ticker(ticker, session=session)
    df = downloader.history(start=date, end=end_date)
    if len(df.index) == 0:
        raise Exception(f"Could not find price for {ticker}")
    price = df.iloc[0]["Close"]

    with open(os.path.join(root_dir, "data", "prices.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ticker, date, price])

if __name__ == "__main__":
    get_price("AAPL", "2018-01-01")