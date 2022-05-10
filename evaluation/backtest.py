"""
Code for evaluating and backtesting the proposed model against various baseline models
"""

import h5py
import os
import numpy as np
from numpy import nan
import yfinance as yf
import pickle
import sys
import tensorflow as tf
from collections import defaultdict
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools

from yfinance.ticker import Ticker

sys.path.append(os.path.join("..", "models"))

from models import model
from models import baseline
from models import RandomModel

DATA_DIR = os

class BackTester:
    def __init__(self, raw_data=False):
        self.raw_data = raw_data
        self.dataset = self.load_h5_dataset()
        self.prices = self.load_prices()

    # Load the .hdf5 dataset into memory
    def load_h5_dataset(self):
        datas = {}
        h5f = h5py.File("../models/parsed_data.hdf5", "r") if not self.raw_data else h5py.File("../models/parsed_data_raw.hdf5", "r")
        for key in h5f.keys():
            datas[key] = eval(h5f[key][()])
        return datas

    def load_prices(self):
        try:
            with open("prices.pickle", "rb") as f:
                return pickle.load(f)
        except (EOFError, FileNotFoundError):
            return {}

    def get_price(self, ticker, year):
        if f"{ticker}-{year}" in self.prices:
            return self.prices[f"{ticker}-{year}"]

        yf_ticker = yf.Ticker(ticker)
        prices = yf_ticker.history(start=f"{year}-12-31", end=f"{year + 1}-01-15")
        try:
            price = prices["Close"].iloc[0]
            if np.isnan(price):
                price = None
        except IndexError:
            price = None

        self.prices[f"{ticker}-{year}"] = price
        print(f"Retrieved price for {ticker}-{year}: {price}")
        with open("prices.pickle", "wb") as f:
            pickle.dump(self.prices, f)
        return price

    # Function for backtesting a model
    def backtest(self, model, test_files, proba=True, buy_threshold=0.33333, sell_threshold=0.33333, from_cache=True, reinvest=False, fin_only=False, mda_only=False):
        years_dict = defaultdict(list)
        for i, file in enumerate(test_files):
            year = int(file[:-7].split("-")[-1])
            years_dict[year].append(i)
        years = sorted(years_dict)

        portfolio = defaultdict(lambda: [0, 0])  # [quantity, avg opening price]
        predictions = model.test_predictions(use_cached=from_cache, fin_only=fin_only, mda_only=mda_only)
        profit_loss_history = []
        buys_history = []
        sells_history = []
        invested = 0
        cash = 0
        for year in years:
            file_idxs = np.array(years_dict[year])
            print(f"Predicting for year: {year}")

            prediction = predictions[file_idxs]
            if proba:
                argmax = tf.squeeze(tf.argmax(prediction, axis=1))
                buys_mask = tf.squeeze(argmax == 2)
                sells_mask = tf.squeeze(argmax == 0)
                buy_condition = tf.logical_and(prediction[:, 2] > buy_threshold, tf.logical_not(sells_mask))
                sell_condition = tf.logical_and(prediction[:, 0] > sell_threshold, tf.logical_not(buys_mask))

                buys_idxs = tf.where(buy_condition).numpy().squeeze()
                sells_idxs = tf.where(sell_condition).numpy().squeeze()
            else:
                buys_idxs = np.where(prediction >= 1)
                sells_idxs = np.where(prediction <= -1)

            buys = file_idxs[buys_idxs]
            sells = file_idxs[sells_idxs]
            if np.isscalar(buys):
                buys = np.array([buys])
            if np.isscalar(sells):
                sells = np.array([sells])

            buys_history.append(len(buys))
            sells_history.append(len(sells))

            # sell
            for idx in sells:
                file = test_files[idx]
                ticker = "".join(file.split("-")[:-1])
                year = int(file[:-7].split("-")[-1])
                if ticker in portfolio and portfolio[ticker] is not None:
                    price = self.get_price(ticker, year)
                    if price is None:
                        continue
                    cash += price * portfolio[ticker][0]
                    portfolio[ticker] = [0, 0]

            # buy
            buys_dict = {}
            for idx in buys:
                file = test_files[idx]
                ticker = "".join(file.split("-")[:-1])
                year = int(file[:-7].split("-")[-1])
                price = self.get_price(ticker, year)
                if price is None:
                    continue
                buys_dict[ticker] = price

            if len(buys_dict.keys()) == 0:
                cash += 1000
            else:
                if reinvest:
                    buy_amount = 1000 + cash
                else:
                    buy_amount = 1000
                for ticker, price in buys_dict.items():
                    if ticker not in portfolio:
                        portfolio[ticker] = [0, 0]
                    position = portfolio[ticker]
                    quantity = (buy_amount / len(buys_dict)) / price
                    new_quantity = quantity + position[0]
                    avg_open = (position[0] * position[1] + quantity * price) / new_quantity
                    portfolio[ticker][0] = new_quantity
                    portfolio[ticker][1] = avg_open
                if reinvest:
                    cash = 0

            profit_loss = cash
            for ticker, value in portfolio.items():
                quantity, avg_open = value
                price = self.get_price(ticker, year)
                if price is None:
                    profit_loss += avg_open * quantity
                else:
                    profit_loss += price * quantity

            invested += 1000
            profit_loss -= invested
            if invested == 0:
                profit_loss_history.append(profit_loss)
            else:
                profit_loss_history.append(profit_loss/invested)

            print(f"profit loss: {profit_loss/invested}")

        profit_loss = cash
        # Sell remaining in portfolio
        for ticker in portfolio:
            if portfolio[ticker] is None:
                continue
            price = self.get_price(ticker, max(list(years_dict.keys())) + 1)
            if price is None:
                profit_loss += portfolio[ticker][0] * portfolio[ticker][1]
            else:
                profit_loss += price * portfolio[ticker][0]

        profit_loss -= invested
        profit_loss_history.append(profit_loss/invested)
        print(profit_loss/invested)
        return profit_loss/invested, profit_loss_history, buys_history, sells_history

    """
    Function to obtain the market performance.
    """
    def backtest_market_avg(self, ticker):
        avg_open = 0
        quantity = 0
        profit_loss_history = []
        invested = 0
        for year in range(2005, 2019):
            price = self.get_price(ticker, year)
            buy_quantity = 1000 / price
            new_quantity = buy_quantity + quantity
            new_avg_open = (1000 + avg_open * quantity) / new_quantity
            quantity = new_quantity
            avg_open = new_avg_open

            invested += 1000
            profit_loss = price * quantity - avg_open * quantity
            if invested == 0:
                profit_loss_history.append(0)
            else:
                profit_loss_history.append(profit_loss/invested)

        price = self.get_price(ticker, 2019)
        profit_loss = price * quantity - avg_open * quantity
        profit_loss_history.append(profit_loss/invested)

        return profit_loss/invested, profit_loss_history

    """
    Function for evaluating a model using statistical metrics
    @param proba :: if predictions are in probabilities or not
    """
    def evaluate(self, model, test_files, proba=True, use_cached=False, fin_only=False, mda_only=False):
        dataset = [self.dataset[file] for file in test_files]
        y = [data["y"] for data in dataset]
        prediction = model.test_predictions(use_cached, fin_only=fin_only, mda_only=mda_only)
        y = y[:len(prediction)]
        if proba:
            y_pred = np.argmax(prediction, axis=-1).squeeze() - 1
        else:
            y_pred = prediction

        for c in [-1, 0, 1]:
            print(f"{c}: {y.count(c)}, {y_pred.tolist().count(c)}")
        print(f"Accuracy: {metrics.accuracy_score(y, y_pred)}")
        print(f"Balanced Accuracy: {metrics.balanced_accuracy_score(y, y_pred)}")
        print(f"micro precision: {metrics.precision_score(y, y_pred, average='micro')}")
        print(f"micro recall: {metrics.recall_score(y, y_pred, average='micro')}")
        print(f"macro precision: {metrics.precision_score(y, y_pred, average='macro')}")
        print(f"macro recall: {metrics.recall_score(y, y_pred, average='macro')}")
        print(f"f1: {metrics.f1_score(y, y_pred, average='macro')}")
        cm = metrics.confusion_matrix(y, y_pred, normalize=None)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 0, 1])
        disp.plot()
        plt.show()
        if proba:
            print(f"auc: {metrics.roc_auc_score(y, prediction, average='macro', multi_class='ovo')}")
            # metrics.RocCurveDisplay.from_predictions(y, y_pred)
            # plt.plot()

# Test the random model 50 times and return the mean
def random_average(model, test_files, reinvest=False):
    profit_losses = []
    histories = []
    buys_history = []
    sells_history = []
    for i in range(50):
        profit_loss, history, buys, sells = backtester.backtest(model, test_files, reinvest=reinvest, proba=False)
        histories.append(history)
        buys_history.append(buys)
        sells_history.append(sells)
        profit_losses.append(profit_loss)

    profit_losses = np.array(profit_losses)
    profit_losses = np.mean(profit_losses, axis=0)
    histories = np.array(histories)
    histories = np.mean(histories, axis=0)
    buys = np.array(buys_history)
    buys = np.mean(buys, axis=0)
    sells = np.array(sells_history)
    sells = np.mean(sells, axis=0)

    return profit_losses, histories, buys, sells


# Plot a cumulative return graph
def plot_history(histories, legends):
    for i, history in enumerate(histories):
        history = np.array(history)
        plt.plot(list(range(2005, 2020)), history * 100, label=legends[i])
        plt.xlabel("year")
    plt.ylabel("cumulative return(%)")


# Plot the number of buys or number of sells as a bar chart
def plot_buys_sells(histories, legends, ylabel="buys"):
    x = np.arange(14)  # the label locations
    width = 0.2  # the width of the bars
    offsets = [-width, 0, width]
    for i, history in enumerate(histories):
        plt.bar(x + offsets[i], history, width, label=legends[i])

    plt.xticks(x, list(range(2005, 2019)))
    plt.xlabel("year")
    plt.ylabel(ylabel)


# Plot the ROC curve
def plot_roc(model, test_files):
    n_classes = 3
    backtester = BackTester(raw_data=True)
    predictions = model.test_predictions(use_cached=True)
    dataset = [backtester.dataset[file] for file in test_files]
    y = [data["y"] for data in dataset]
    for i, y_true in enumerate(y):
        y[i] = [1 if y_true == x else 0 for x in [-1, 0, 1]]
    y = np.array(y)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], predictions[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y.ravel(), predictions.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    lw = 2
    colors = itertools.cycle(["red", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i-1, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.show()

# Plot precision-recall graph
def plot_precision_recall(model, test_files):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = 3
    backtester = BackTester(raw_data=True)
    predictions = model.test_predictions(use_cached=True)
    dataset = [backtester.dataset[file] for file in test_files]
    y = [data["y"] for data in dataset]
    y_labels = y[:]
    for i, y_true in enumerate(y):
        y[i] = [1 if y_true == x else 0 for x in [-1, 0, 1]]
    y = np.array(y)

    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y[:, i], predictions[:, i])
        average_precision[i] = metrics.average_precision_score(y[:, i], predictions[:, i])

    _, ax = plt.subplots(figsize=(7, 8))

    colours = ["blue", "green", "orange"]
    for i in range(n_classes):
        display = metrics.PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i-1}", color=colours[i])
        ax.axhline(y=y_labels.count(i-1)/len(y_labels), linestyle=":", color=colours[i])

    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend()

    plt.show()

# Plot the heatmap of the NN model at different buy and sell thresholds
def test_thresholds(model, test_files):
    num = 12
    buy_thresholds = np.linspace(0.2, 0.65, num)
    sell_thresholds = np.linspace(0, 1.0, num)
    results = np.zeros((num, num))
    backtester = BackTester(raw_data=True)
    for i, bt in enumerate(buy_thresholds):
        for j, st in enumerate(sell_thresholds):
            results[i][j] = backtester.backtest(model, test_files, buy_threshold=bt, sell_threshold=st, reinvest=True, from_cache=True, proba=True)[0]

    results *= 100
    results = np.round(results.T).astype(int)

    fig, ax = plt.subplots()
    im = ax.imshow(results, cmap="bwr_r")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(buy_thresholds)), labels=np.round(buy_thresholds, decimals=2))
    ax.set_yticks(np.arange(len(sell_thresholds)), labels=np.round(sell_thresholds, decimals=2))

    # Loop over data dimensions and create text annotations.
    for i in range(len(buy_thresholds)):
        for j in range(len(sell_thresholds)):
            ax.text(j, i, results[i, j], ha="center", va="center", color="k")

    ax.invert_yaxis()
    ax.set_title("Cumulative profit(%)")
    ax.set_xlabel("buy threshold")
    ax.set_ylabel("sell threshold")
    fig.tight_layout()
    plt.show()

# Function to plot the returns of NN model at different thresholds
def plot_threshold_returns(model, test_files, thresholds):
    for threshold in thresholds:
        backtester = BackTester(raw_data=True)
        _, history, _, _ = backtester.backtest(model, test_files, buy_threshold=threshold[0], sell_threshold=threshold[1], reinvest=True, from_cache=True, proba=True)
        plt.plot(list(range(2005, 2020)), np.array(history)*100, label=f"buy:{threshold[0]}, sell:{threshold[1]}")


if __name__ == "__main__":
    nn_model = model.load_model("../models/512_1_.49acc")  # Load the best NN model

    # Load the baseline and random model
    baseline_model = baseline.Baseline()
    random_model = RandomModel.RandomModel()

    # Prepare the back tester
    test_files = nn_model.test_files
    backtester = BackTester(raw_data=True)

    # Change these values to backtest using different configurations
    reinvest = True  # To backtest with reinvestment or not
    fin_only = False  # To backtest using just the financials or not
    mda_only = False  # To backtest using just the MD&A or not

    # Get the result of backtesting
    sandp_pl, sandp_histories = backtester.backtest_market_avg("^GSPC")
    random_pl, random_histories, random_buys, random_sells = random_average(random_model, test_files, reinvest=reinvest)
    baseline_pl, baseline_histories, baseline_buys, baseline_sells = backtester.backtest(baseline_model, test_files, proba=False, reinvest=reinvest, from_cache=True)
    nn_pl, nn_histories, nn_buys, nn_sells = backtester.backtest(nn_model, test_files, reinvest=reinvest, from_cache=True)

    print(sandp_pl, random_pl, baseline_pl, nn_pl)

    # Plot the graph for cummulative return for the different models
    plot_history([sandp_histories, random_histories, baseline_histories, nn_histories], ["S&P500", "random", "baseline", "NN model"])

    # Uncomment below to plot the return graph for ideal threshold
    # plot_history([sandp_histories, random_histories], ["S&P500", "random"])
    # plot_threshold_returns(nn_model, test_files, thresholds=[(0.45, 0.61), (0.33, 0.33)])

    plt.legend()
    plt.show()

    # Test the statistical metrics for NN model and baseline model
    backtester.evaluate(nn_model, test_files, proba=True, use_cached=True, fin_only=fin_only, mda_only=mda_only)
    backtester.evaluate(baseline_model, test_files, proba=False, use_cached=True)

    ## Other available functions ##
    # plot_roc(nn_model, test_files)
    # plot_precision_recall(nn_model, test_files)
    # test_thresholds(nn_model, test_files)

    # print(baseline_buys, nn_buys)
    # plot_buys_sells([random_buys, baseline_buys, nn_buys], ["random", "baseline", "NN model"], "buys")









