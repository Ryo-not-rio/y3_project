"""
Code for processing the data into .pickle files
"""

import os
import pandas as pd
import pickle
import yfinance as yf

import alpha_vantage as av

# Get the year from a date
def date_to_year(date):
    if date.month <= 6:
        year = date.year - 1
    else:
        year = date.year

    return year


def get_financial(ticker, year, df):
    df = df[(df["ticker"] == ticker) & (df["year"] == year)]
    df = df.drop(columns=['ticker', 'year', 'index'])
    df = df / df.iloc[0]["market capitalization"]
    df = df.drop(columns=["market capitalization"])
    return df.iloc[0].to_dict()


def get_avg(year):
    with open(os.path.join("..", "data", "financial_avgs.csv")) as f:
        df = pd.read_csv(f, index_col=[0])

    df = df.loc[year]
    df = df / df["market capitalization"]
    df = df.drop(columns=["market capitalization"])

    return df.to_dict()


def get_price(ticker, year):
    yf_ticker = yf.Ticker(ticker)
    prices = yf_ticker.history(start=f"{year}-12-31", end=f"{year + 1}-01-15")
    return prices["Close"].iloc[0]


def get_price_av(ticker, year, av_obj):
    try:
        return av_obj.get_price(ticker, year)
    except:
        return None


def get_avg_price(year):
    df = pd.read_csv(os.path.join("..", "data", "avg_prices.csv"), index_col=[0])
    return df['price'].loc[year]


def get_y(ticker, year, y_ahead, av_obj, threshold=0.1):
    # Get price using alpha vantage
    # ticker_current = get_price_av(ticker, year, av_obj)
    # ticker_future = get_price_av(ticker, year+y_ahead, av_obj)

    # Get price using yahoo
    ticker_current = get_price(ticker, year)
    ticker_future = get_price(ticker, year+y_ahead)
    if ticker_current is None or ticker_future is None:
        return None
    ticker_change = (ticker_future-ticker_current)/ticker_current

    avg_current = get_avg_price(year)
    avg_future = get_avg_price(year+y_ahead)
    avg_change = (avg_future-avg_current)/avg_current

    if ticker_change-avg_change > threshold:
        return 1
    if ticker_change-avg_change < -threshold:
        return -1
    return 0


def save_data(ticker, year, data_dict):
    with open(os.path.join("..", "data", "parsed_data2", f"{ticker}-{year}.pickle"), "wb") as f:
        pickle.dump(data_dict, f)


# Main code for processing the data
def construct_ds(y_ahead=3):
    with open(os.path.join("..", "data_gathering", "active.csv")) as f:
        df = pd.read_csv(f)

    with open(os.path.join("..", "data", "financials_w_marketcap.csv")) as f:
        fin_df = pd.read_csv(f)

    av_obj = av.AlphaVantage()
    fin_df = fin_df[fin_df['market capitalization'].notnull()] # Filter datapoints where market cap is too small
    ds_path = os.path.join("..", "data", "mda")

    # Need a dictionary of {"mda", "y-2fin", "y-1fin", "y0fin", "y-2_avg_fin", "y-1_avg_fin", "y0_avg_fin", "y"}
    for i, row in df.iterrows():
        ticker = row["symbol"]
        if ticker <= "KNX":
            continue
        for year in range(2005, 2022-y_ahead):
            print(f"Constructing data for {ticker} - {year}")
            data = {}
            # Get mda
            file = os.path.join(ds_path, f"{ticker}_{year}.txt")
            if not os.path.isfile(file):
                continue
            else:
                with open(file, encoding='utf-8') as f:
                    data["mda"] = f.read()

            print("Gathered mda")

            # Get fins
            try:
                fin0 = get_financial(ticker, year, fin_df)
                fin_1 = get_financial(ticker, year-1, fin_df)
                fin_2 = get_financial(ticker, year-2, fin_df)
            except IndexError:
                continue
            else:
                data["y0fin"] = fin0
                data["y-1fin"] = fin_1
                data["y-2fin"] = fin_2

            # Get market avg fins
            data["y0_avg_fin"] = get_avg(year)
            data["y-1_avg_fin"] = get_avg(year-1)
            data["y-2_avg_fin"] = get_avg(year-2)
            print("Gathered fins & avgs")

            # Get y
            try:
                y = get_y(ticker, year, y_ahead, av_obj)
                if y is None:
                    continue
                data["y"] = y
            except IndexError:
                continue
            print("Gathered y")

            save_data(ticker, year, data)
            print("...finished and saved.\n")


if __name__ == "__main__":
    construct_ds()
