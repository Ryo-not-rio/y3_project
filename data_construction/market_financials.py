import os
import pandas as pd
from collections import defaultdict


def save_market_avgs():
    with open(os.path.join("..", "data", "financials_w_marketcap.csv")) as f:
        df = pd.read_csv(f)

    avgs = pd.DataFrame()
    for year in range(2003, 2019):
        _df = df[df['year'] == year].sort_values('market capitalization', 0, ascending=False)
        top_500 = _df.head(500)
        print(top_500)
        top_500 = top_500.drop(['ticker', 'index', 'year'], 1)
        sum = top_500['market capitalization'].sum()
        mult_arr = top_500['market capitalization']/sum
        # print(mult_arr)
        top_500 = top_500.multiply(mult_arr, 0)
        summed = top_500.sum(axis=0)
        summed.name = year
        avgs = avgs.append(summed)
        # avgs.loc[year] = summed

    with open(os.path.join("..", "data", "financial_avgs.csv"), "w") as f:
        avgs.to_csv(f)


if __name__ == "__main__":
    save_market_avgs()

