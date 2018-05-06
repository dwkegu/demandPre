import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import BasicLSTMCell


class SCLSTM(RNNCell):

    def __init__(self, input_shape, output_shape, activation, name=None):
        super(RNNCell, self).__init__(name=name)

    @property
    def state_size(self):
        pass

    @property
    def output_size(self):
        pass

    def build(self, _):
        super().build(_)

    def zero_state(self, batch_size, dtype):
        return super().zero_state(batch_size, dtype)

