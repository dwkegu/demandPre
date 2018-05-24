import tensorflow as tf
import numpy as np
from demandPre.src.model.model import Model


class CubeCNN(Model):

    def __init__(self, input_shape, output_shape, name="CubeCNN"):
        super(CubeCNN, self).__init__(input_shape, output_shape, model_name=name)

    def block_A(self, input, params, activation=tf.nn.sigmoid, name="blockA"):
        '''
        周期性patten提取
        :param activation:
        :param input: [Batch_size, heig
        :param params:[[12, 5], [24, 10], [24*7, 20]]
        :param name:
        :return:
        '''
        assert len(input.shape) == 5
        nets = []
        with tf.variable_scope(name, default_name="blockA"):
            nets.clear()
            for i, param in enumerate(params):
                kernel = tf.get_variable("kernel_T%d" % i, shape=[param[0], 1, 1, input.shape[4], input.shape[4]], dtype=np.float32)
                bias = tf.get_variable("bias_T%d" % i, shape=[input.shape[4]], dtype=np.float32)
                net = tf.nn.conv3d(input, kernel, [1, param[1], 1, 1, 1], 'VALID', name=name+"conv3d_%d" % i)
                net = tf.nn.bias_add(net, bias)
                net = activation(net)
                kerne1_1 = tf.get_variable("kernel_T%d_1" % i, shape=[net.shape[1], 1, 1, input.shape[4], param[1]], dtype=np.float32)
                bias1_1 = tf.get_variable("bias_T%d_1" % i, shape=[param[1]], dtype=np.float32)
                net = tf.nn.conv3d(net, kerne1_1, [1, 1, 1, 1, 1], 'VALID', name=name+"conv3d_%d_1" % i)
                net = tf.nn.bias_add(net, bias1_1)
                net = activation(net)
                #[None, 1, height, width, param[1]]
                net = tf.reshape(net, [-1, net.shape[2], net.shape[3], net.shape[4]])
                nets.append(net)
            net = tf.concat(nets, axis=3)
        return net

    def block_B(self, inputs, params, activation=tf.nn.sigmoid, name="blockB"):
        '''

        :param inputs:
        :param params: [[[5, 5], [1, 1, 1, 1], 8],]
        :param activation:
        :param name:
        :return:
        '''
        assert len(inputs.shape) == 4
        net = inputs
        with tf.variable_scope(name):
            for i, param in enumerate(params):
                kernel = tf.get_variable("kernel_%d" % i, param[0], dtype=tf.float32)
                bias = tf.get_variable("bias_%d" % i, param[2], dtype=tf.float32)
                net = tf.nn.conv2d(net, kernel, param[1], 'SAME')
                net = tf.nn.bias_add(net, bias)
                net = activation(net)
        net = tf.reshape(net, [-1, net.shape[1], net.shape[2]])
        return net

    def build(self,):
        with tf.name_scope("layer1"):
            net = self.block_A(self._inputs, [[1, 5, 8], [1, 9, 8], [1, 12, 8], [1, 24, 8]], activation=tf.nn.relu, name="layer1_blockA")
        with tf.name_scope("layer2"):
            net = self.block_B(net, [[[5, 5, 32, 16], [1, 1, 1, 1], 16], [[3, 3, 16, 8], [1, 1, 1, 1], 8], [[3, 3, 8, 1], [1, 1, 1, 1], 1]],
                               activation=tf.nn.relu, name="layer2_blockB")
        net = tf.reshape(net, [-1, 1, net.shape[1], net.shape[2], 1])
        loss = 2 * tf.nn.l2_loss(net-self._outputs)
        self._loss = loss
        global_steps = tf.Variable(0, dtype=tf.int32, trainable=False)
        train_op = tf.train.RMSPropOptimizer(self._lnr).minimize(self._loss, global_step=global_steps)
        self._train_op = train_op
        print(self.get_num_params())
        self._built = True