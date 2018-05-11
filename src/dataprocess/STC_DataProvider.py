from demandPre.src.dataprocess.dataProvider import DataProvider
import numpy as np
from demandPre.src.dataprocess import nyprocess
from demandPre.src.dataprocess import bjprocessor


class STC_Provider(DataProvider):

    def __init__(self, filenames, t_length, batch_size, input_size, output_size, splits=None,
                 train_proprotion=(0.8, 0.1, 0.1)):
        super(STC_Provider, self).__init__(filenames, batch_size, input_size, output_size)
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
        self.t_length = t_length
        self.data_offset = t_length * input_size + self._output_size - 1
        self.hasValidData = True
        if splits is None:
            if len(train_proprotion) == 3:
                self.hasValidData = True
            else:
                self.hasValidData = False
            self.time_length = self.data.shape[0]
            self.train_length = int(self.time_length * train_proprotion[0])
            self.valid_length = int(self.time_length * train_proprotion[1])
            self.test_length = self.time_length - self.train_length - self.valid_length
            self.train_data = self.data[0:self.train_length, :, :, :]
            self.valid_data = self.data[self.train_length:self.train_length + self.valid_length, :, :, :]
            self.test_data = self.data[self.train_length + self.valid_length:self.time_length, :, :, :]
        else:
            if len(splits) == 3:
                self.hasValidData = True
                self.train_length = splits[0]
                self.valid_length = splits[1]
                self.test_length = splits[2]
                self.train_data = self.data[0:splits[0], :, :, :]
                self.valid_data = self.data[splits[0] - self.data_offset:splits[0] + splits[1], :, :, :]
                self.test_data = self.data[splits[0] + splits[1] - self.data_offset:splits[0] + splits[1] + splits[2],
                                 :, :, :]
            elif len(splits) == 2:
                self.hasValidData = False
                self.train_length = splits[0]
                self.test_length = splits[1]
                self.train_data = self.data[0:splits[0], :, :, :, :]
                self.test_data = self.data[splits[0] - self.data_offset:splits[0] + splits[1], :, :, :, :]
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
            s = np.sum(np.power(self.train_data, 2))
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

    def get_train_batch2(self):
        train_counter = 0
        for _data in self.train_data:
            position = 0
            while position + self._input_size * self.t_length + self._output_size - 1 < _data.shape[0]:
                if position + self._batch_size + self._input_size * self.t_length + self._output_size - 1 >= \
                        _data.shape[0]:
                    x = []
                    y = []
                    for i in range(position,
                                   _data.shape[0] - self._input_size * self.t_length - self._output_size + 1):
                        example_x = []
                        for j in range(i, i + self._input_size * self.t_length,
                                       self._input_size):
                            sample_x = _data[j:j + self._input_size, :, :, :]
                            example_x.append(sample_x)
                        example_x = np.stack(example_x, axis=0)
                        example_y = _data[
                                    i + self._input_size * self.t_length:i + self._input_size * self.t_length + self._output_size,
                                    :, :, 0:1]
                        # example_y = np.expand_dims(example_y, axis=0)
                        x.append(example_x)
                        y.append(example_y)
                    position = _data.shape[0] - self._input_size - self._output_size + 1
                    # print(np.array(x).shape)
                    # print(np.array(y).shape)
                    train_counter += len(x)
                    yield (x, y)
                else:
                    x = []
                    y = []
                    for i in range(position, position + self._batch_size):
                        example_x = []
                        for j in range(i, i + self._input_size * self.t_length, self._input_size):
                            sample_x = _data[j:j + self._input_size, :, :, :]
                            example_x.append(sample_x)
                        example_x = np.stack(example_x, axis=0)
                        example_y = _data[
                                    i + self._input_size * self.t_length:i + self._input_size * self.t_length + self._output_size,
                                    :, :, 0:1]
                        # example_y = np.expand_dims(example_y, axis=0)
                        x.append(example_x)
                        y.append(example_y)
                    position += self._batch_size
                    # print(np.array(x).shape)
                    # print(np.array(y).shape)
                    train_counter += len(x)
                    yield (x, y)
            print(train_counter)

    def get_train_epoch_size2(self):
        print("%d x %d x %d" % (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
        s = 0
        for _data in self.train_data:
            s += (_data.shape[0] - self._input_size * self.t_length - self._output_size + 1) * _data.shape[1] * \
                 self.data.shape[2] * (self.data.shape[3] - 1)
        print(s)
        return s

    def get_train_batch(self):
        position = 0
        train_counter = 0
        while position + self._input_size * self.t_length + self._output_size - 1 < self.train_length:
            if position + self._batch_size + self._input_size * self.t_length + self._output_size - 1 >= self.train_length:
                x = []
                y = []
                for i in range(position, self.train_length - self._input_size * self.t_length - self._output_size + 1):
                    example_x = []
                    for j in range(i, i + self._input_size * self.t_length,
                                   self._input_size):
                        sample_x = self.train_data[j:j + self._input_size, :, :, :]
                        example_x.append(sample_x)
                    example_x = np.stack(example_x, axis=0)
                    example_y = self.train_data[
                                i + self._input_size * self.t_length:i + self._input_size * self.t_length + self._output_size,
                                :, :, 0:1]
                    # example_y = np.expand_dims(example_y, axis=0)
                    x.append(example_x)
                    y.append(example_y)
                position = self.train_length - self._input_size - self._output_size + 1
                # print(np.array(x).shape)
                # print(np.array(y).shape)
                train_counter += len(x)
                yield (x, y)
            else:
                x = []
                y = []
                for i in range(position, position + self._batch_size):
                    example_x = []
                    for j in range(i, i + self._input_size * self.t_length, self._input_size):
                        sample_x = self.train_data[j:j + self._input_size, :, :, :]
                        example_x.append(sample_x)
                    example_x = np.stack(example_x, axis=0)
                    example_y = self.train_data[
                                i + self._input_size * self.t_length:i + self._input_size * self.t_length + self._output_size,
                                :, :, 0:1]
                    # example_y = np.expand_dims(example_y, axis=0)
                    x.append(example_x)
                    y.append(example_y)
                position += self._batch_size
                # print(np.array(x).shape)
                # print(np.array(y).shape)
                train_counter += len(x)
                yield (x, y)
        print("train len is %d" % train_counter)

    def get_train_epoch_size(self):
        print("%d x %d x %d" % (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
        return (self.train_length - self._input_size * self.t_length - self._output_size + 1) * self.data.shape[1] * \
               self.data.shape[2] * (self.data.shape[3] - 1)

    def get_valid_batch(self):
        position = 0
        valid_count = 0
        while position < self.valid_length:
            if position + self._batch_size >= self.valid_length:
                x = []
                y = []
                for i in range(position, self.valid_length):
                    example_x = []
                    for j in range(i, i + self._input_size * self.t_length,
                                   self._input_size):
                        sample_x = self.valid_data[j:j + self._input_size, :, :, :]
                        example_x.append(sample_x)
                    example_x = np.stack(example_x, axis=0)
                    example_y = self.valid_data[
                                i + self._input_size * self.t_length:i + self._input_size * self.t_length + self._output_size,
                                :, :, 0:1]
                    # example_y = np.expand_dims(example_y, axis=0)
                    x.append(example_x)
                    y.append(example_y)
                position = self.valid_length
                # print(np.array(x).shape)
                # print(np.array(y).shape)
                valid_count += len(x)
                yield (x, y)
            else:
                x = []
                y = []
                for i in range(position, position + self._batch_size):
                    example_x = []
                    for j in range(i, i + self._input_size * self.t_length, self._input_size):
                        sample_x = self.valid_data[j:j + self._input_size, :, :, :]
                        example_x.append(sample_x)
                    example_x = np.stack(example_x, axis=0)
                    example_y = self.valid_data[
                                i + self._input_size * self.t_length:i + self._input_size * self.t_length + self._output_size,
                                :, :, 0:1]
                    # example_y = np.expand_dims(example_y, axis=0)
                    x.append(example_x)
                    y.append(example_y)
                position += self._batch_size
                # print(np.array(x).shape)
                # print(np.array(y).shape)
                valid_count += len(x)
                yield (x, y)
        print("valid data len is %d" % valid_count)

    def get_valid_epoch_size(self):
        print("%d x %d x %d" % (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
        return self.valid_length * self.data.shape[1] * self.data.shape[2] * (self.data.shape[3] - 1)

    def get_test_batch(self):
        position = 0
        while position < self.test_length:
            if position + self._batch_size >= self.test_length:
                # print(self._input_size)
                x = []
                y = []
                for i in range(position, self.test_length):
                    example_x = []
                    for j in range(i, i + self._input_size * self.t_length, self._input_size):
                        sample_x = self.test_data[j:j + self._input_size, :, :, :]
                        example_x.append(sample_x)
                    example_x = np.stack(example_x, axis=0)
                    example_y = self.test_data[
                                i + self._input_size * self.t_length:i + self._input_size * self.t_length + self._output_size,
                                :, :, 0:1]
                    # example_y = np.expand_dims(example_y, axis=0)
                    x.append(example_x)
                    y.append(example_y)
                position = self.test_length
                yield (x, y)
            else:
                x = []
                y = []
                for i in range(position, position + self._batch_size):
                    example_x = []
                    for j in range(i, i + self._input_size * self.t_length, self._input_size):
                        sample_x = self.test_data[j:j + self._input_size, :, :, :]
                        example_x.append(sample_x)
                    example_x = np.stack(example_x, axis=0)
                    example_y = self.test_data[
                                i + self._input_size * self.t_length:i + self._input_size * self.t_length + self._output_size,
                                :, :, 0:1]
                    # example_y = np.expand_dims(example_y, axis=0)
                    x.append(example_x)
                    y.append(example_y)
                position += self._batch_size
                yield (x, y)

    def get_test_epoch_size(self):
        print("%d x %d x %d" % (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
        return self.test_length * self.data.shape[1] * self.data.shape[2] * (self.data.shape[3] - 1)
