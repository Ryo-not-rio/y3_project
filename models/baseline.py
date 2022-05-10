"""
Implementation of the baseline model.
Code includes fitting the data to the model and testing the model.
For detailed evaluation of the model, use code in ../evaluation
"""

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import svm, pipeline, feature_selection
from collections import defaultdict
import numpy as np
import joblib
import pickle
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import common

DATA_DIR = os.path.join("..", "data", "parsed_data")
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Baseline:
    def __init__(self):
        # Dictionary of {tfidfs, "y-2fin", "y-1fin", "y0fin", "y-2_avg_fin", "y-1_avg_fin", "y0_avg_fin", "y"}
        self.datas = defaultdict(list)
        self.train_files, self.test_files = common.train_test_files(raw_data=True)
        self.svm = svm.SVC(max_iter=10000)
        self.vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.90, stop_words="english")
        self.feature_selector = None

    def dataset_from_files(self, files):
        dataset = []
        for file in files:
            with open(os.path.join(DATA_DIR, file), "rb") as f:
                data = pickle.load(f)
                dataset.append(data)
        return dataset

    def process_dataset(self, dataset, balance_data=True, new_scaler=False):
        mdas = []
        datas = defaultdict(list)
        for data in dataset:
            if "market capitalization" not in data["y0fin"].keys():
                continue

            mdas.append(data['mda'])

            for k, v in data.items():
                if k != "mda":
                    if type(v) is dict:
                        v = list(v.values())
                    datas[k].append(v)

        datas["tfidfs"] = self.vectorizer.fit_transform(mdas).toarray()

        for k, v in datas.items():
            if k not in ["tfidfs", "y"]:
                arr = np.array(v)
                if new_scaler or not os.path.isfile("baseline_scaler.pickle"):
                    scaler = StandardScaler()
                else:
                    with open("baseline_scaler.pickle", "rb") as f:
                        scaler = pickle.load(f)
                datas[k] = scaler.fit_transform(arr)

                if new_scaler or not os.path.isfile("baseline_scaler.pickle"):
                    with open("baseline_scaler.pickle", "wb") as f:
                        pickle.dump(scaler, f)

        if balance_data:
            y_count = {}
            for y in set(datas["y"]):
                y_count[y] = datas["y"].count(y)

            sum_count = sum(list(y_count.values()))
            for y, count in y_count.items():
                matching_idx = [i for i, x in enumerate(datas["y"]) if x == y]
                if count > sum_count/3:
                    for k in datas.keys():
                        datas[k] = np.delete(datas[k], matching_idx[:int(count - sum_count/3)], axis=0)
                else:
                    for _ in range(int(sum_count/3 - count)):
                        idx = random.choice(matching_idx)
                        for k, v in datas.items():
                            datas[k] = np.append(datas[k], [v[idx]], axis=0)

        return datas

    def process_files(self, files, balance_data=True, new_scaler=False):
        dataset = self.dataset_from_files(files)
        return self.process_dataset(dataset, balance_data, new_scaler)

    def construct_data(self, data_points=1000, force=False, balance_data=True):
        if force or not os.path.isfile("datas.sav"):
            print("Gathering and saving data into datas.sav")
            files = self.train_files
            random.shuffle(files)
            if data_points != "max":
                files = files[:data_points]
            self.datas = self.process_files(files, new_scaler=True, balance_data=balance_data)

            print("Finished gathering. Saving data...")
            with open("datas.sav", "wb") as f:
                joblib.dump(self.datas, f, compress=0)
        else:
            with open("datas.sav", "rb") as f:
                self.datas = joblib.load(f)
        print("Loaded data")
        return self.datas

    # Save the model
    def save(self):
        with open(os.path.join(DIR_PATH, "baseline.sav"), "wb") as f:
            joblib.dump({"tfidf vectorizer": self.vectorizer, "feature selector": self.feature_selector, "model": self.svm}, f)

    # Load cached model
    def load(self):
        with open(os.path.join(DIR_PATH, "baseline.sav"), "rb") as f:
            baseline_dict = joblib.load(f)
        self.vectorizer = baseline_dict["tfidf vectorizer"]
        self.feature_selector = baseline_dict["feature selector"]
        self.svm = baseline_dict["model"]

    # Fit onto data
    def fit(self, colab=False, save=True, k=5000):
        print("Preparing data")
        datas = self.datas
        if not self.datas:
            datas = self.construct_data()

        X = np.concatenate([v for k, v in datas.items() if k != "y"], axis=1)
        X = np.nan_to_num(X)
        y = datas["y"]
        self.feature_selector = pipeline.Pipeline([
            ('low variance', feature_selection.VarianceThreshold()),
            ('k best', feature_selection.SelectKBest(feature_selection.mutual_info_classif, k=k))
        ])

        pipe = pipeline.Pipeline([
            ('feature selection', self.feature_selector),
            ('classification', self.svm)
        ])

        print("Fitting SVM")
        pipe.fit(X, y)

        print("Testing SVM")
        if colab:
            random.shuffle(self.test_files)
            self.test_files = self.test_files[:500]

        if not os.path.exists("baseline_test_data.pkl"):
            datas_test = self.process_files(self.test_files, balance_data=False)
            X_test = np.concatenate([v for k, v in datas_test.items() if k != "y"], axis=1)
            X_test = np.nan_to_num(X_test)
            y_test = datas_test["y"]
            with open("baseline_test_data.pkl", "wb") as f:
                pickle.dump((X_test, y_test), f)
        else:
            with open("baseline_test_data.pkl", "rb") as f:
                X_test, y_test = pickle.load(f)

        X_test = self.feature_selector.fit_transform(X_test, y_test)
        accuracy = self.svm.score(X_test, y_test)
        print("accuracy: ", accuracy)
        if save:
            self.save()
        return accuracy

    # Test the model on the test set and cache the results
    def save_test_predictions(self):
        self.load()
        dataset = self.dataset_from_files(self.test_files)
        datas = self.process_dataset(dataset, balance_data=False)
        X = np.concatenate([v for k, v in datas.items() if k != "y"], axis=1)
        X = np.nan_to_num(X)
        X = self.feature_selector.transform(X)
        print("predicting...")
        predictions = self.svm.predict(X)
        with open(os.path.join(DIR_PATH, "test_predictions_baseline.pkl"), "wb") as f:
            pickle.dump(predictions, f)

        return predictions

    def test_predictions(self, use_cached=False, **kwargs):
        if use_cached:
            with open(os.path.join(DIR_PATH, "test_predictions_baseline.pkl"), "rb") as f:
                predictions = pickle.load(f)
        else:
            predictions = self.save_test_predictions()
        return predictions

    # Test different number of ks for feature selection
    def test_different_ks(self, ks):
        results = []
        for k in ks:
            result = self.fit(k=k, save=False)
            results.append(result)
        print(results)
        plt.plot(ks, results)
        print(results)
        plt.xlabel("K")
        plt.ylabel("accuracy")
        plt.xscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    obj = Baseline()
    obj.test_different_ks([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3000, 5120, 6000, 8000, 9000])
    obj.fit(k=6000)  # Fit the model on k=6000
    obj.test_predictions(use_cached=True)  # Test the model on test data