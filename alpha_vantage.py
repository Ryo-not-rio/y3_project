"""
Code for downloading price from Alpha Vantage. Backup in case yfinance is not working.
"""

import requests
from datetime import datetime, timedelta
import time

# API_KEY = "OT2J6KH7T43LXL3F"
API_KEY = "16NEF91YTS4L8RK3"
BASE_URL = "https://www.alphavantage.co/query"

class AlphaVantage:
    def __init__(self):
        self.cache = {}

    def get_price(self, ticker, year, retry=False):
        if ticker in self.cache:
            series = self.cache[ticker]
        else:
            try:
                url = f"{BASE_URL}?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={ticker}&apikey={API_KEY}"
                r = requests.get(url)
                data = r.json()
                series = data['Monthly Adjusted Time Series']
                self.cache[ticker] = series
            except Exception as e:
                print(f"An error occurred: {e}")
                if not retry:
                    print("Retrying...")
                    time.sleep(60)
                    return self.get_price(ticker, year, retry=True)
                else:
                    return None

        search_date = datetime.fromisoformat(f"{year}-12-31")
        for k, v in series.items():
            if datetime.fromisoformat(k) - search_date <= timedelta(31):
                return float(v['5. adjusted close'])
