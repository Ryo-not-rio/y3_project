"""
Random model. Only used in the backtesting stage
"""

import numpy as np

import common


class RandomModel:
    def __init__(self, weights=(1, 1, 1)):
        self.weights=weights

    def test_predictions(self, **kwargs):
        _, test_files = common.train_test_files(raw_data=True)
        p = [w/sum(self.weights) for w in self.weights]
        predictions = [np.random.choice([-1, 0, 1], p=p) for _ in range(len(test_files))]
        return np.array(predictions)