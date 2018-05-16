import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple

t_Conv = "t_conv"
s_Conv = "s_conv"


class STC_LSTMCell(RNNCell):

    def __init__(self, input_shape, output_shape, forget_bias=0.0, params=None, activation=tf.nn.relu, name=None):
        '''

        :param input_shape:
        :param output_shape:
        :param params:
        :param activation:
        :param name:
        '''
        # super(RNNCell, self).__init__(name=name)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._activation = activation
        self._name = name
        self._forget_bias = forget_bias
        if params is None:
            # t_size t_stride channel_size
            # params = [[[12, 12, 12], [24, 24, 24]], [[1, 5, 5, 36, output_shape[-1]], [1, 1, 1, 1, 1]]]
            params = [[[4, 4, 8], [6, 6, 12], [8, 8, 16], [12, 12, 24], [24, 24, 48]], [[1, 5, 5, 108, output_shape[-1]], [1, 1, 1, 1, 1]]]
        self._params = params
        self.conv_variables = {}

    @property
    def state_size(self):
        return LSTMStateTuple(self.output_shape, self.output_shape)

    @property
    def output_size(self):
        return self._output_shape

    def zero_state(self, batch_size, dtype):
        output_shape = [batch_size]
        output_shape.extend(self._output_shape)
        c_zeros = tf.zeros(output_shape, dtype=dtype)
        h_zeros = tf.zeros(output_shape, dtype=dtype)
        zeros = tf.contrib.rnn.LSTMStateTuple(c_zeros, h_zeros)
        return zeros

    def __call__(self, inputs, status):
        c, h = status
        net = inputs
        nets = []
        if inputs is not None:
            with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
                for i in range(len(self._params)):
                    if i == 0:
                        nets_f = []
                        nets_i = []
                        nets_j = []
                        nets_o = []
                        params = self._params[i]
                        for j in range(len(params)):
                            param = params[j]
                            kernel1 = tf.get_variable("t_kernel_%d_%d_0" % (i, j),
                                                      [param[0], 1, 1, self._input_shape[4], param[2] * 4],
                                                      dtype=tf.float32)
                            bias1 = tf.get_variable("t_bias_%d_%d_0" % (i, j), [param[2] * 4], dtype=tf.float32)
                            net = tf.nn.conv3d(inputs, kernel1, [1, param[1], 1, 1, 1], 'VALID')
                            net = tf.nn.bias_add(net, bias1)
                            # todo activation
                            net = self._activation(net)
                            kernel2 = tf.get_variable("t_kernel_%d_%d_1" % (i, j),
                                                      [net.shape[1], 1, 1, param[2] * 4, param[2] * 4],
                                                      dtype=tf.float32)
                            net = tf.nn.conv3d(net, kernel2, [1, 1, 1, 1, 1], 'VALID')
                            bias2 = tf.get_variable("t_bias_%d_%d_1" % (i, j), [param[2] * 4], dtype=tf.float32)
                            net = tf.nn.bias_add(net, bias2)
                            # todo activation
                            f, ii, jj, o = tf.split(net, 4, axis=4)
                            nets_f.append(f)
                            nets_i.append(ii)
                            nets_j.append(jj)
                            nets_o.append(o)
                        net_f = tf.concat(nets_f, axis=4)
                        net_i = tf.concat(nets_i, axis=4)
                        net_j = tf.concat(nets_j, axis=4)
                        net_o = tf.concat(nets_o, axis=4)
                        net = [net_f, net_i, net_j, net_o]
                    else:
                        params = self._params[i]
                        k_size = params[0]
                        nets = []
                        for j in range(4):
                            # new_input = tf.concat([net[j]], axis=4)
                            kernel = tf.get_variable("s_kernel_%d_%d_f" % (i, j),
                                                     [k_size[0], k_size[1], k_size[2], k_size[3], k_size[4]],
                                                     dtype=tf.float32)
                            bias = tf.get_variable("s_bias_%d_%d_f" % (i, j), [k_size[-1]], dtype=tf.float32)
                            net_ = tf.nn.conv3d(net[j], kernel, params[1], 'SAME')
                            net_ = tf.nn.bias_add(net_, bias)
                            nets.append(net_)
            f, i, j, o = nets
        else:
            f = i = j = o = 0
        # with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
        #     h_w_f = tf.get_variable("h_w_f", shape=self._output_shape,
        #                             dtype=tf.float32)
        #     h_w_i = tf.get_variable("h_w_i", shape=self._output_shape,
        #                             dtype=tf.float32)
        #     h_w_j = tf.get_variable("h_w_j", shape=self._output_shape,
        #                             dtype=tf.float32)
        #     h_w_o = tf.get_variable("h_w_o", shape=self._output_shape,
        #                             dtype=tf.float32)
        # f = f + tf.multiply(h, h_w_f[tf.newaxis, :, :, :, :])
        # i = i + tf.multiply(h, h_w_i[tf.newaxis, :, :, :, :])
        # j = j + tf.multiply(h, h_w_j[tf.newaxis, :, :, :, :])
        # o = o + tf.multiply(h, h_w_o[tf.newaxis, :, :, :, :])
        sigmoid = tf.nn.sigmoid
        new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j)
        # new_x = o
        o = self._activation(o)
        new_x = tf.concat([o, new_c], axis=4)
        with tf.variable_scope(self.name + "g_h", reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable("kernel", [1, 1, 1, new_x.shape[4], self._output_shape[-1]])
            bias = tf.get_variable("bias", [self._output_shape[-1]])
            new_h = tf.nn.conv3d(new_x, kernel, [1, 1, 1, 1, 1], 'SAME')
            new_h = tf.nn.bias_add(new_h, bias)
            new_h = self._activation(new_h)
        # new_h = sigmoid(o) * self._activation(new_c)
        new_state = LSTMStateTuple(new_c, new_h)
        return new_h, new_state
