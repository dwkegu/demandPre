import sys
import platform
import os
import numpy as np
windows_project_path = "e:/PyCharm/"
linux_project_path = "/home/pengshunfeng/pyCharmProject/"
platform_string = platform.platform()
project_path = windows_project_path if platform_string.__contains__("indow") else linux_project_path
sys.path.append(project_path)
# from dataprocess import didiprocess
from demandPre.src import config
from demandPre.src.model.CubeCNN import CubeCNN
from demandPre.src.dataprocess.dataProvider import DidiDataProvider
from demandPre.src.dataprocess.STC_DataProvider import STC_Provider
from demandPre.src.model.STC_LSTM import STC_Lstm

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >' + config.log_path + '/tmp')
memory_gpu=[int(x.split()[2]) for x in open(config.log_path + '/tmp', 'r').readlines()]
print(memory_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))

if __name__ == '__main__':
    #[batch, T, d, h, w c]
    model = STC_Lstm([None, 7, 24, 64, 64, 1], [None, 1, 64, 64, 1], learning_rate=0.00002, name="B96-STC=LSTM", normalize=True)
    # filenames = os.listdir(config.dataset_path)
    # files = ["BJ13_M32x32_T30_InOut.h5", "BJ14_M32x32_T30_InOut.h5", "BJ15_M32x32_T30_InOut.h5", "BJ16_M32x32_T30_InOut.h5"]
    # allFiles = [os.path.join(config.dataset_path, file) for file in files]
    # print(allFiles)
    # dataset = STC_Provider(filenames=allFiles, t_length=7, batch_size=48, input_size=48, output_size=1, splits=[10248, 1128, 1344])
    # dataset = STC_Provider(config.dataset_path + "/NYC14_M16x8_T60_NewEnd.h5", 7, 16, 24, 1, [3737, 415, 240])
    dataset = STC_Provider(config.dataset_path + "/nyt_d_map.mat", 7, 24, 24, 1, [14640, 1440, 1440], offset=False, normalize=True)
    model.fit(dataset, 100)
