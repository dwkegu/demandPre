import tensorflow as tf
from demandPre.src.model.model import Model
from demandPre.src.model.STC_LSTMCell import STC_LSTMCell
from tensorflow.python.ops import array_ops


class STC_Lstm(Model):

    def __init__(self, input_shape, output_shape, name="STC-LSTM"):
        '''

        :param input_shape: [batch_size, lstm_t, cnn_t, height, width, channel]
        :param output_shape: [batch_size, cnn_t, height, width, channel]
        :param name:
        '''
        super(STC_Lstm, self).__init__(input_shape, output_shape, name)
        self.lstm_output_shape = output_shape[1:]
        # self._batch_size = batch_size

    def build(self):
        self.cell = STC_LSTMCell(self._input_shape, self.lstm_output_shape, activation=tf.nn.relu, name="STC_LSTM")
        lstm_t = self._input_shape[1]
        first_input = self._inputs[0]
        input_shape = first_input.get_shape().with_rank_at_least(2)
        fixed_batch_size = input_shape[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(first_input)[0]
        state = self.cell.zero_state(batch_size, dtype=tf.float32)
        y = None
        for i in range(lstm_t):
            x = self._inputs[:, i, :, :, :, :]
            y, state = self.cell(x, state)
        if y is None:
            raise ValueError("input list T is less than 1")
        self._loss = 2 * tf.nn.l2_loss(y - self._outputs)
        self._train_op = tf.train.RMSPropOptimizer(self._lnr).minimize(self._loss)
        self._built = True