import numpy as np
import pandas as pd


class DataProvider:

    def __init__(self, filenames, batch_size, input_size, output_size):
        self._filename = filenames
        self._batch_size = batch_size
        self._input_size = input_size
        self._output_size = output_size

    def get_train_batch(self):
        pass

    def get_train_epoch_size(self):
        pass

    def get_test_batch(self):
        pass

    def get_test_epoch_size(self):
        pass


class DidiDataProvider(DataProvider):

    def __init__(self, filenames, batch_size, input_size, output_size):
        super(DidiDataProvider, self).__init__(filenames, batch_size, input_size, output_size)
        if filenames.endswith('.npy'):
            self.data = np.load(filenames)
            self.data = np.transpose(self.data, [2, 1, 0])
            self.data = np.expand_dims(self.data, 3)
        self.time_length = self.data.shape[0]
        self.train_length = int(self.time_length * 0.8)
        self.test_length = self.time_length - self.train_length
        self.train_data = self.data[0:self.train_length, :, :, :]
        self.test_data = self.data[self.train_length:self.time_length, :, :, :]
        s = np.sum(np.power(self.train_data, 2))
        count = np.prod(self.train_data.shape)
        s = np.sqrt(s/count)
        print("average of train is %f" % s)
        s = np.sum(np.power(self.test_data, 2))
        count = np.prod(self.test_data.shape)
        s = np.sqrt(s / count)
        print("average of test is %f" % s)

    def get_train_batch(self):
        position = 0
        while position + self._input_size + self._output_size - 1 < self.train_length:
            if position + self._batch_size + self._input_size + self._output_size - 1 >= self.train_length:
                x = []
                y = []
                for i in range(position, self.train_length - self._input_size - self._output_size + 1):
                    sample_x = self.train_data[position:position+self._input_size, :, :, :]
                    sample_y = self.train_data[position+self._input_size:position+self._input_size+self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position = self.train_length - self._input_size - self._output_size + 1
                yield (x, y)
            else:
                x = []
                y = []
                for i in range(position, self._batch_size + position):
                    sample_x = self.train_data[position:position + self._input_size, :, :, :]
                    sample_y = self.train_data[
                               position + self._input_size:position + self._input_size + self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position += self._batch_size
                yield (x, y)

    def get_train_epoch_size(self):
        return (self.train_length - self._input_size - self._output_size + 1) * 64 * 64

    def get_test_batch(self):
        position = 0
        while position + self._input_size + self._output_size - 1 < self.test_length:
            if position + self._batch_size + self._input_size + self._output_size - 1 >= self.test_length:
                x = []
                y = []
                for i in range(position, self.test_length - self._input_size - self._output_size + 1):
                    sample_x = self.test_data[position:position + self._input_size, :, :, :]
                    sample_y = self.test_data[
                               position + self._input_size:position + self._input_size + self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position = self.test_length - self._input_size - self._output_size + 1
                yield (x, y)
            else:
                x = []
                y = []
                for i in range(position, self._batch_size + position):
                    sample_x = self.test_data[position:position + self._input_size, :, :, :]
                    sample_y = self.test_data[
                               position + self._input_size:position + self._input_size + self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position += self._batch_size
                yield (x, y)

    def get_test_epoch_size(self):
        return (self.test_length - self._input_size - self._output_size + 1) * 64 * 64
