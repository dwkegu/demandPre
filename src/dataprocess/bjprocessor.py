import numpy as np
import os
import h5py
import datetime
from demandPre.src import config

def load_bj_data(filenames):
    data = []
    timeStamps = []
    for filename in filenames:
        file = h5py.File(filename, 'r')
        # print(file['date'].shape)
        # print(file['data'].shape)
        data.append(file['data'])
        timeStamps.append(file['date'])
    data = np.concatenate(data, axis=0)
    data = np.transpose(data, [0, 1, 2, 3])
    timeStamps = np.concatenate(timeStamps, axis=0)
    print(timeStamps.shape)
    last_time = None
    hours = []
    invalidRow = []
    lastValid = 0
    for i, item in enumerate(timeStamps):
        # print(str(item, encoding='ascii'))
        time = datetime.datetime.strptime(str(item, encoding='ascii')[:-2], "%Y%m%d")
        if last_time is None or time == last_time:
            hours.append(int(item[-2:]))
        else:
            if len(hours) < 48:
                if i - len(hours) - lastValid > 24 * 7:
                    invalidRow.append((lastValid, i - len(hours)))
                lastValid = i
            hours.clear()
            hours.append(int(item[-2:]))
        last_time = time
    new_data = []
    print(invalidRow)
    for i in invalidRow:
        new_data.append(data[i[0]:i[1]])
    # new_data = np.concatenate(new_data, axis=0)
    # print(new_data.shape)
    return new_data


filepath = config.dataset_path
filenames = os.listdir(filepath)
filenames = [os.path.join(filepath, file) for file in filenames if file.endswith("InOut.h5")]
load_bj_data(filenames)
