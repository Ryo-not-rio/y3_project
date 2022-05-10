import yfinance as yf
import pandas as pd
import os

if __name__ == "__main__":
    prices_list = []
    for year in range(2003, 2022):
        yf_ticker = yf.Ticker("^GSPC")
        try:
            price = yf_ticker.history(start=f"{year}-12-31", end=f"{year + 1}-01-15")['Close'].iloc[0]
        except:
            continue
        prices_list.append(price)

    df = pd.DataFrame(prices_list, index=list(range(2003, 2022)), columns=['price'])
    df.to_csv(os.path.join("..", "data", "avg_prices.csv"))