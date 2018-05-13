import os
import csv
import numpy as np
from demandPre.src import config
import math


def get_3d_demand_data():
    '''
    obtain the dataset
    :return:
    '''
    order_files = os.listdir(config.cd_didi_order_path)
    ltLng = config.CD_LT_LONG
    ltLat = config.CD_LT_LAT
    rbLng = config.CD_RB_LONG
    rbLat = config.CD_RB_LAT
    latStep = config.CD_LAT_STEP
    lngStep = config.CD_LONG_STEP
    timeUnit = config.CD_TIME_UNIT
    timeNum = int(math.ceil(config.CD_ENDTIME -config.CD_STARTTIME) / timeUnit)
    demandMap = np.ndarray([32, 32, timeNum])
    for file in order_files:
        print("next")
        with open(config.cd_didi_order_path + os.path.sep +file, 'r', encoding='utf8') as f:
            f_csv = csv.reader(f)
            for line in f_csv:
                time1 = int(line[1])
                time2 = int(line[2])
                long_start = float(line[3])
                lat_start = float(line[4])
                if long_start < ltLng or lat_start > ltLat or long_start > rbLng or lat_start < rbLat:
                    continue
                i = int((ltLat - lat_start) / latStep)
                j = int((long_start - ltLng) / lngStep)
                t = int((time1 - config.CD_STARTTIME) / timeUnit)
                demandMap[i, j, t] += 1
    print(config.dataset_path)
    np.save(config.dataset_path + os.path.sep + "demand_map", demandMap)


# get_3d_demand_data()
# a = np.array([[1, 2],[3, 4]])
# print(np.sum(a))
# demandMap = np.load(config.dataset_path + os.path.sep + "demand_map.npy")
# print(np.max(demandMap))
# print(demandMap.shape)
# s = np.sum(np.power(demandMap, 2))
# print(np.sqrt(s/np.prod(demandMap.shape)))
# total_loss = 0
# for i in range(24*4, demandMap.shape[2]):
#     hi = [j for j in range(i-24*4, -1, -24*4)]
#     hd = demandMap[:, :, hi]
#     ha = np.sum(hd, axis=2)
#     hc = hd.shape[2]
#     ha = ha/hc
#     loss = np.sum(np.power(demandMap[:, :, i] - ha, 2))
#     total_loss += loss
# total_loss /= ((demandMap.shape[2] - 24*4) * demandMap.shape[0] * demandMap.shape[1])
# print(total_loss)