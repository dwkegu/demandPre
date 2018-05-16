import scipy.io as sio
import numpy as np
import h5py
from demandPre.src import config
import pylab


def load_data(filename):
    if len(filename) == 2:
        d1 = sio.loadmat(filename[0])['p_map']
        d2 = sio.loadmat(filename[1])['d_map']
        data = np.concatenate((d1[:, :, :, np.newaxis], d2[:, :, :, np.newaxis]), axis=3)
    else:
        data = sio.loadmat(filename)['d_map']
    # t_length = data.shape[0]
    # train_length = int(t_length * split)
    # train = data[0:train_length, :, :, :]
    # test = data[train_length:, :, :, :]
    print(data.shape)
    return data[-17520:, :, :]


def load_nyb_data(filename):
    data = h5py.File(filename, 'r')
    data = data['data'][:, 0:1, :, :]
    data = np.transpose(data, [0, 2, 3, 1])
    return data


# load_nyb_data(config.dataset_path + "/NYC14_M16x8_T60_NewEnd.h5")
# data = load_data(config.dataset_path + "/nyt_d_map.mat")
# s = np.sum(np.power(data[-17520: - 17520 + 12000], 2))
# print(np.sum(data[-17520 + 24]))
# print(s)
# ac = np.sqrt(s/np.prod(data[0:12000].shape))
# print(ac)
# dt = [12, 24, 24 * 7, 24 * 7 * 2]
# x = []
# y = []
# i = 20
# while i < 24 * 30 * 4 + 20:
#     x.append(i)
#     y.append(data[i, 22, 26])
#     i += 1
# pylab.plot(x, y)
# pylab.show()