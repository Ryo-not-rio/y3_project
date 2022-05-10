"""
Contains functions for analysing the scraped data.
To analyse the data, run each function according to what data needs to be analysed.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("..", "data")

# Show the average lengths of the MD&A gathered
def mda_lengths():
    lengths = []
    mda_dir = os.path.join(DATA_DIR, "mda")
    for file in os.listdir(mda_dir):
        with open(os.path.join(mda_dir, file), encoding='utf-8') as f:
            r = f.read()
        lengths.append(len(r.split(" ")))

    lengths = np.array(lengths)
    print(f"Median: {np.median(lengths)}, Mean: {np.mean(lengths)}, Min: {np.min(lengths)}, Max: {np.max(lengths)}")

# Show the number of MD&A documents gathered and the number of companies in the data gathered
def mda_num():
    mda_dir = os.path.join(DATA_DIR, "mda")
    files = os.listdir(mda_dir)
    num = len(files)
    companies = set([file.split("_")[0] for file in files])
    print(f"Number of mdas: {num}, Number of companies: {len(companies)}")

# Show the number of financials gathered and the number of companies that make up the data
def financials_num():
    mda_dir = os.path.join(DATA_DIR, "mda")
    mda_files = os.listdir(mda_dir)
    fin_file = os.path.join(DATA_DIR, "financials_w_marketcap.csv")
    df = pd.read_csv(fin_file)
    count = 0
    companies = {}
    for file in mda_files:
        ticker, year = file.split("_")[:2]
        year = year[:-4]
        match = df[(df["ticker"] == ticker) & (df["year"] == int(year))]
        if len(match.index > 0):
            count += 1
            companies[ticker] = True
    print(f"Number of data points: {count}, Number of companies: {len(companies)}")

# Show the number of data points and companies after market cap filteration is applied
def filtered_num():
    fin_file = os.path.join(DATA_DIR, "financials_w_marketcap.csv")
    df = pd.read_csv(fin_file)
    df = df[df['market capitalization'].notnull()]
    print(f"Number of data points: {len(df.index)}, Number of companies: {len(set(df['ticker'].tolist()))}")

# Show the number of data points and companies in the final dataset
def final_num(files=None):
    if files is None:
        final_dir = os.path.join(DATA_DIR, "parsed_data")
        files = os.listdir(final_dir)
    companies = set([file.split("-")[0] for file in files])
    print(f"Number of data points: {len(files)}, Number of companies: {len(companies)}")

# Show the number of data points and number of companies in the test dataset
def test_num():
    with open("../models/train_test_split_raw.csv") as f:
        train_str, test_str = f.readlines()

    test_files = test_str.split(",")
    final_num(test_files)


if __name__ == "__main__":
    mda_lengths()
    # financials_num()
    # filtered_num()
    # final_num()
    # test_num()
