import numpy as np


class Normalor:

    def __init__(self, data):
        self.max_item = np.max(data)
        self.min_item = np.min(data)
        self.data = data
        print("max item is %d min item is %d" % (self.max_item, self.min_item))

    def fit(self):
        new_data = (self.data - self.min_item) / (self.max_item - self.min_item)
        return new_data

    def restoreLoss(self, loss):
        true_loss = loss * (self.max_item - self.min_item)
        return true_loss