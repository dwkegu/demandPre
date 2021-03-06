import h5py
import numpy as np
import os
from demandPre.src import config

def read_data(filename):
    print(filename)
    data = h5py.File(filename, 'r')
    _data = data['data'][:, 0:1, :, :]
    print(_data.shape)
    return np.transpose(_data, [0, 2, 3, 1])


# read_data(os.path.join(config.dataset_path, "NYC14_M16x8_T60_NewEnd.h5"))