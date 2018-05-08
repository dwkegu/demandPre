import scipy.io as sio
import numpy as np
import h5py
from demandPre.src import config


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
    return data


def load_nyb_data(filename):
    data = h5py.File(filename, 'r')
    print(data['data'].shape)
    data = data['data'][:, :, :, :]
    data = np.transpose(data, [0, 2, 3, 1])
    return data


load_nyb_data(config.dataset_path + "/NYC14_M16x8_T60_NewEnd.h5")