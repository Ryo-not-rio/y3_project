"""
Scrape the market cap for a company by scraping their price and multiplying it by the number of shared outstanding
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import requests
from pandas_datareader import data
import math

import alpha_vantage as av

root_dir = "D:\\PycharmProjects\\y3_project"
hdr = 'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'

def get_market_cap_av(ticker, year, av_obj):
    try:
        current_market_cap = data.get_quote_yahoo(ticker)['marketCap'].iloc[0]
        if current_market_cap < 1e9:
            print("Current market cap is too low")
            return None
        prev_price = av_obj.get_price(ticker, year)
        current_price = av_obj.get_price(ticker, 2021)
    except Exception as e:
        print(f"Could not retrieve from api: {e}")
        return None

    if prev_price is None or current_price is None:
        return None

    return current_market_cap * (prev_price/current_price)


def get_market_cap_yf(ticker, year, session):
    try:
        current_market_cap = data.get_quote_yahoo(ticker)['marketCap'].iloc[0]
        if current_market_cap < 1e9:
            print("Current market cap is too low")
            return None
        yf_ticker = yf.Ticker(ticker)
        prev_prices = yf_ticker.history(start=f"{year}-12-31", end=f"{year + 1}-01-15", session=session)
        current_prices = yf_ticker.history(period='5d', session=session)
    except Exception as e:
        print(f"Could not retrieve from api: {e}")
        return None

    if prev_prices.empty or current_prices.empty:
        print("Could not retrieve either the previous prices or the current prices")
        return None

    if prev_prices.iloc[0].name - datetime.fromisoformat(f"{year}-12-31") > timedelta(days=180):
        print("Previous price is not within the acceptable range")
        return None
    return current_market_cap * (prev_prices['Close'].iloc[0]/current_prices['Close'].iloc[0])


def scrape_market_cap():
    df_path = os.path.join(root_dir, "data", "financials_w_marketcap.csv")
    df = pd.read_csv(df_path, low_memory=False, memory_map=True)
    if "market capitalization" not in df:
        df["market capitalization"] = None

    session = requests.Session()
    session.headers['User-agent'] = hdr

    av_obj = av.AlphaVantage()

    count = 0
    for idx, row in df.iterrows():
        if "CLB" <= row["ticker"] <= "E":
            print(f"Getting market cap for {row['ticker']} - {row['year']}")
            market_cap = get_market_cap_av(row["ticker"], row["year"], av_obj)
            # market_cap = get_market_cap_yf(row["ticker"], row["year"], session)
            df.at[idx, "market capitalization"] = market_cap
            count += 1
            if count >= 50:
                print("Writing to file...")
                df.to_csv(df_path, index=False)
                count = 0
                session = requests.Session()
                session.headers['User-agent'] = hdr
                time.sleep(2)

if __name__ == "__main__":
    scrape_market_cap()
    # print(get_market_cap("wtm", 2018))