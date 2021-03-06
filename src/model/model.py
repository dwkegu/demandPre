import tensorflow as tf
import numpy as np
import time
import os
from demandPre.src import config
from functools import reduce
from operator import mul


class Model:
    def __init__(self, input_shape, output_shape, learning_rate=0.0002, model_name="LSTM", normalize=False):
        self._model_name = model_name
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._loss = None
        self._train_op = None
        self._y = None
        self._inputs = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name="inputs")
        self._outputs = tf.placeholder(dtype=tf.float32, shape=self._output_shape, name="outputs")
        self._built = False
        self._lnr = learning_rate
        self._model_path = os.path.join(config.log_path, model_name)
        self._normalize = normalize

    def build(self):
        pass

    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    def fit(self, dataset, epoches=50):
        gpu_opt = tf.ConfigProto()
        gpu_opt.gpu_options.allow_growth = True
        if not self._built:
            self.build()
        initial = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        if self._loss is None or self._train_op is None:
            raise ValueError("loss or train_op is None")
        with tf.Session(config=gpu_opt) as sess:
            saver = tf.train.Saver(tf.global_variables())
            summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(os.path.join(self._model_path, "model"), sess.graph)
            sess.run([initial])
            min_valid_score = 100
            best_model_index = 92
            for i in range(epoches):
                save = True
                start_time = time.time()
                train_data = dataset.get_train_batch()
                total_loss = 0
                for t_x, t_y in train_data:
                    [loss, _] = sess.run([self._loss, self._train_op],
                                            feed_dict={self._inputs: t_x, self._outputs: t_y})
                    total_loss += loss
                    # if save:
                    #     np.save(os.path.join(config.log_path, "output", "output-%d.npy" % i), y)
                    #     np.save(os.path.join(config.log_path, "output", "label-%d.npy" % i), t_y)
                    #     save = False
                rmse = np.sqrt(total_loss / dataset.get_train_epoch_size())
                if self._normalize:
                    rmse = dataset._normalor.restoreLoss(rmse)
                print("training epoch %d, loss is %f, rmse is %f" % (
                    i, total_loss, rmse))
                if dataset.hasValidData:
                    valid_data = dataset.get_valid_batch()
                    total_loss = 0
                    for t_x, t_y in valid_data:
                        [loss] = sess.run([self._loss], feed_dict={self._inputs: t_x, self._outputs: t_y})
                        total_loss += loss
                        # print(loss)
                    now = time.time()
                    valid_rmse = np.sqrt(total_loss / dataset.get_valid_epoch_size())
                    if self._normalize:
                        valid_rmse = dataset._normalor.restoreLoss(valid_rmse)
                    print("\ntime is %ds valid rmse is %f " % (now - start_time, valid_rmse))
                    if valid_rmse < min_valid_score:
                        path = saver.save(sess, os.path.join(self._model_path, "saved_model"), global_step=i + 1)
                        print(path)
                        print("model-%s saved." % (i + 1))
                        best_model_index = i + 1
                        min_valid_score = valid_rmse
                else:
                    test_data = dataset.get_test_batch()
                    total_loss = 0
                    for t_x, t_y in test_data:
                        [loss] = sess.run([self._loss], feed_dict={self._inputs: t_x, self._outputs: t_y})
                        total_loss += loss
                    test_rmse = np.sqrt(total_loss / dataset.get_test_epoch_size())
                    now = time.time()
                    if self._normalize:
                        test_rmse = dataset._normalor.restoreLoss(test_rmse)
                    print("time si %d test rmse is %f " % ((now - start_time), test_rmse))
                if i % 5 == 0:
                    if dataset.hasValidData:
                        # saver.restore(sess, self._model_path + "-" + str(best_model_index))
                        test_data = dataset.get_test_batch()
                        total_loss = 0
                        for t_x, t_y in test_data:
                            [loss] = sess.run([self._loss], feed_dict={self._inputs: t_x, self._outputs: t_y})
                            total_loss += loss
                        test_rmse = np.sqrt(total_loss / dataset.get_test_epoch_size())
                        if self._normalize:
                            test_rmse = dataset._normalor.restoreLoss(test_rmse)
                        print("test rmse is %f " % test_rmse)
            if dataset.hasValidData:
                saver.restore(sess, os.path.join(self._model_path, "saved_model-") + str(best_model_index))
                test_data = dataset.get_test_batch()
                total_loss = 0
                for t_x, t_y in test_data:
                    [loss] = sess.run([self._loss], feed_dict={self._inputs: t_x, self._outputs: t_y})
                    total_loss += loss
                test_rmse = np.sqrt(total_loss / dataset.get_test_epoch_size())
                if self._normalize:
                    test_rmse = dataset._normalor.restoreLoss(test_rmse)
                print("test rmse is %f " % test_rmse)
