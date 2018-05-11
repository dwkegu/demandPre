import numpy as np
import os
import h5py
import datetime
import calendar
import pylab
from mpl_toolkits.mplot3d import Axes3D
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
    data = np.transpose(data, [0, 2, 3, 1])
    new_data = []
    last_week_day = None
    hours = []
    last_time = None
    new_timeStamps = []
    timeStamps = np.concatenate(timeStamps, axis=0)
    print(timeStamps.shape)
    skip_day = 0
    for i, t_item in enumerate(timeStamps):
        now = datetime.datetime.strptime(str(t_item, encoding='ascii')[:-2], "%Y%m%d")
        if last_time is None:
            last_time = now
        weekday = now.weekday()
        if last_time is None or now == last_time:
            hours.append(int(t_item[-2:]))
        else:
            if len(hours) == 48 and (last_week_day is None or weekday == (last_week_day + 1) % 7):
                new_data.append(data[i - 48:i])
                new_timeStamps.append(last_time.strftime("%Y%m%d"))
                last_week_day = weekday
            if len(hours) < 48:
                print("%s %d  %d" % (now.strftime("%Y%m%d"), last_week_day, weekday))
            skip_day += 1
            hours.clear()
            hours.append(int(t_item[-2:]))
            last_time = now
    # print(skip_day)
    # print(len(new_timeStamps))
    new_data = np.concatenate(new_data, axis=0)
    print(new_data.shape)
    return new_data


# filepath = config.dataset_path
# filenames = os.listdir(filepath)
# filenames = [os.path.join(filepath, file) for file in filenames if file.endswith("InOut.h5")]
# data = load_bj_data(filenames)

# plotData = data[0]


# def f_map(x, y):
#     z = np.ndarray(shape=x.shape, dtype=np.float32)
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             z[x[i, j], y[i, j]] = plotData[12, x[i, j], y[i, j], 0]
#     return z
#
#
# y = plotData[:, 8, 8, 0]
# x = range(plotData.shape[0])
# pylab.plot(x, y)
# pylab.show()
