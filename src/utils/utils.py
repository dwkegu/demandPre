import numpy as np


class Normalor:

    def __init__(self, data):
        self.max_item = np.max(data)
        self.min_item = np.min(data)
        self.data = data

    def fit(self):
        new_data = (self.data - self.min_item)/(2*self.max_item)
        return new_data

    def restoreLoss(self, loss):
        true_loss = None