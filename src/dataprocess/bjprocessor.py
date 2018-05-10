import numpy as np
import os
import h5py
from demandPre.src import config

def load_bj_data(filenames):
    data = []
    for filename in filenames:
        file = h5py.File(filename, 'r')
        print(file['date'].shape)
        print(file['data'].shape)
        data.append(file['data'])
    data = np.concatenate(data, axis=0)
    return data


# filepath = config.dataset_path
# filenames = os.listdir(filepath)
# filenames = [os.path.join(filepath, file) for file in filenames if file.endswith("InOut.h5")]
# load_bj_data(filenames)