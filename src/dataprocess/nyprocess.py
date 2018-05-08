import scipy.io as sio
import numpy as np
import h5py


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
    data = data['data'][:, 0:1, :, :]
    data = np.transpose(data, [0, 2, 3, 1])
    return data