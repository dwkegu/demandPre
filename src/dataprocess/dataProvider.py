import numpy as np
import pandas as pd
from demandPre.src.dataprocess import nyprocess, bjprocessor
from demandPre.src.utils.utils import Normalor


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


class CNNDataProvider(DataProvider):

    def __init__(self, filenames, batch_size, input_size, output_size, splits=None, output_reduce_channel=1,
                 train_proprotion=None, offset=True, normalize=False):
        super(CNNDataProvider, self).__init__(filenames, batch_size, input_size, output_size)
        if train_proprotion is None:
            train_proprotion = [0.8, 0.1, 0.1]
        self._output_reduce_channel = output_reduce_channel
        if isinstance(filenames, str) and filenames.endswith('.npy'):
            self.data = np.load(filenames)
            self.data = np.transpose(self.data, [2, 1, 0])
            self.data = np.expand_dims(self.data, 3)
        elif isinstance(filenames, str) and filenames.endswith('mat'):
            self.data = nyprocess.load_data(filenames)
            self.data = np.expand_dims(self.data, 3)
        elif isinstance(filenames, str) and filenames.endswith(".h5"):
            self.data = nyprocess.load_nyb_data(filenames)
        elif isinstance(filenames, (list, tuple)):
            self.data = bjprocessor.load_bj_data(filenames)
        if normalize:
            self._normalor = Normalor(self.data)
            self.data = self._normalor.fit()
        self.data_offset = input_size + self._output_size - 1
        self.hasValidData = True
        if splits is None:
            if len(train_proprotion) == 3:
                self.hasValidData = True
            else:
                self.hasValidData = False
            self.time_length = self.data.shape[0]
            self.train_length = int(self.time_length * train_proprotion[0])
            self.train_data = self.data[0:self.train_length, :, :, :]
            if offset:
                self.valid_length = int(self.time_length * train_proprotion[1])
                self.test_length = self.time_length - self.train_length - self.valid_length
                self.valid_data = self.data[self.train_length - self.data_offset:self.train_length + self.valid_length,
                                  :, :, :]
                self.test_data = self.data[self.train_length + self.valid_length - self.data_offset:self.time_length, :,
                                 :, :]
            else:
                self.valid_length = int(self.time_length * train_proprotion[1]) - self.data_offset
                self.test_length = self.time_length - self.train_length - self.valid_length - self.data_offset
                self.valid_data = self.data[self.train_length:self.train_length + self.valid_length, :, :, :]
                self.test_data = self.data[self.train_length + self.valid_length:self.time_length, :, :, :]
        else:
            if len(splits) == 3:
                self.hasValidData = True
                self.train_length = splits[0]
                self.valid_length = splits[1] if offset else splits[1] - self.data_offset
                self.test_length = splits[2] if offset else splits[2] - self.data_offset
                print(self.train_length)
                print(self.valid_length)
                print(self.test_length)
                self.train_data = self.data[0:splits[0], :, :, :]
                if offset:
                    self.valid_data = self.data[splits[0] - self.data_offset:splits[0] + splits[1], :, :, :]
                    self.test_data = self.data[
                                     splits[0] + splits[1] - self.data_offset:splits[0] + splits[1] + splits[2],
                                     :, :, :]
                else:
                    self.valid_data = self.data[splits[0]:splits[0] + splits[1], :, :, :]
                    self.test_data = self.data[
                                     splits[0] + splits[1]:splits[0] + splits[1] + splits[2],
                                     :, :, :]
            elif len(splits) == 2:
                self.hasValidData = True
                self.valid_length = splits[0]
                self.test_length = splits[1]
                _data = self.data[-1]
                self.valid_data = _data[-(splits[0] + splits[1] + self.data_offset):, :, :, :, :]
                self.test_data = _data[- (self.test_length + self.data_offset):, :, :, :, :]
                for i, item in enumerate(self.data):
                    if i == len(self.data):
                        self.train_data.append(item[:-(splits[0] + splits[1])])
                    else:
                        self.train_data.append(item)
                self.data = np.concatenate(self.data)
            elif len(splits) == 1 and isinstance(self.data, (list, tuple)):
                self.hasValidData = False
                self.test_data = self.data[-1]
                self.test_length = splits[0]
                self.test_data = self.test_data[-(splits[0] + self.data_offset):]
                self.train_data = []
                for i, item in enumerate(self.data):
                    if i == len(self.data):
                        self.train_data.append(item[:splits[0]])
                    else:
                        self.train_data.append(item)
                self.data = np.concatenate(self.data)
        if isinstance(self.train_data, np.ndarray):
            s = np.sum(np.power(self.train_data, 2), dtype=np.int64)
            count = np.prod(self.train_data.shape)
            s = np.sqrt(s / count)
            print("average of train is %f" % s)
            s = np.sum(np.power(self.valid_data, 2))
            count = np.prod(self.valid_data.shape)
            s = np.sqrt(s / count)
            print("average of valid is %f" % s)
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
                    sample_x = self.train_data[i:i+self._input_size, :, :, :]
                    sample_y = self.train_data[i+self._input_size:i+self._input_size+self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position = self.train_length - self._input_size - self._output_size + 1
                yield (x, y)
            else:
                x = []
                y = []
                for i in range(position, self._batch_size + position):
                    sample_x = self.train_data[i:i + self._input_size, :, :, :]
                    sample_y = self.train_data[
                               i + self._input_size:i + self._input_size + self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position += self._batch_size
                yield (x, y)

    def get_train_epoch_size(self):
        return (self.train_length - self._input_size - self._output_size + 1) * self.data.shape[1] * self.data.shape[2] * (self.data.shape[3] - self._output_reduce_channel)

    def get_valid_batch(self):
        position = 0
        while position + self._input_size + self._output_size - 1 < self.valid_length:
            if position + self._batch_size + self._input_size + self._output_size - 1 >= self.valid_length:
                x = []
                y = []
                for i in range(position, self.valid_length - self._input_size - self._output_size + 1):
                    sample_x = self.valid_data[i:i + self._input_size, :, :, :]
                    sample_y = self.valid_data[
                               i + self._input_size:i + self._input_size + self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position = self.valid_length - self._input_size - self._output_size + 1
                yield (x, y)
            else:
                x = []
                y = []
                for i in range(position, self._batch_size + position):
                    sample_x = self.valid_data[i:i + self._input_size, :, :, :]
                    sample_y = self.valid_data[
                               i + self._input_size:i + self._input_size + self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position += self._batch_size
                yield (x, y)

    def get_valid_epoch_size(self):
        return (self.valid_length - 0 if self.data_offset else (self._input_size - self._output_size + 1))\
               * self.data.shape[1] * self.data.shape[2] * (self.data.shape[3] - self._output_reduce_channel)

    def get_test_batch(self):
        position = 0
        while position + self._input_size + self._output_size - 1 < self.test_length:
            if position + self._batch_size + self._input_size + self._output_size - 1 >= self.test_length:
                x = []
                y = []
                for i in range(position, self.test_length - self._input_size - self._output_size + 1):
                    sample_x = self.test_data[i:i + self._input_size, :, :, :]
                    sample_y = self.test_data[
                               i + self._input_size:i + self._input_size + self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position = self.test_length - self._input_size - self._output_size + 1
                yield (x, y)
            else:
                x = []
                y = []
                for i in range(position, self._batch_size + position):
                    sample_x = self.test_data[i:i + self._input_size, :, :, :]
                    sample_y = self.test_data[
                               i + self._input_size:i + self._input_size + self._output_size,
                               :, :, :]
                    x.append(sample_x)
                    y.append(sample_y)
                position += self._batch_size
                yield (x, y)

    def get_test_epoch_size(self):
        return (self.test_length - 0 if self.data_offset else (self._input_size - self._output_size + 1))\
               * self.data.shape[1] * self.data.shape[2] * (self.data.shape[3]- self._output_reduce_channel)
