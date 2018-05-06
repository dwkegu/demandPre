import tensorflow as tf
from demandPre.src.model.model import Model
from demandPre.src.model.STC_LSTMCell import STC_LSTMCell
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn import _best_effort_input_batch_size
import numpy as np


class STC_Lstm(Model):

    def __init__(self, input_shape, output_shape, name="STC-LSTM"):
        '''

        :param input_shape: [batch_size, lstm_t, cnn_t, height, width, channel]
        :param output_shape: [batch_size, cnn_t, height, width, channel]
        :param name:
        '''
        super(STC_Lstm, self).__init__(input_shape, output_shape, model_name=name)
        self.lstm_output_shape = output_shape[1:]
        # self._batch_size = batch_size

    def build(self):
        cell_input_shape = self._input_shape.copy()
        cell_input_shape.pop(1)
        self.cell = STC_LSTMCell(cell_input_shape, self.lstm_output_shape, activation=tf.nn.relu, name="STC_LSTMCell")
        lstm_t = self._input_shape[1]
        flat_input = tf.transpose(self._inputs, [1, 0, 2, 3, 4, 5])
        flat_input = nest.flatten(flat_input)
        batch_size = _best_effort_input_batch_size(flat_input)
        state = self.cell.zero_state(batch_size, dtype=tf.float32)
        y = None
        for i in range(lstm_t):
            x = self._inputs[:, i, :, :, :, :]
            y, state = self.cell(x, state)
        # y, states = tf.nn.dynamic_rnn(self.cell, self._inputs)
        if y is None:
            raise ValueError("input list T is less than 1")
        self._loss = 2 * tf.nn.l2_loss(y - self._outputs)
        self._train_op = tf.train.RMSPropOptimizer(self._lnr).minimize(self._loss)
        print(self.get_num_params())
        self._built = True