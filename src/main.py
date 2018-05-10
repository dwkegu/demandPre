import sys
import platform
import os

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

if __name__ == '__main__':
    #[batch, T, d, h, w c]
    model = STC_Lstm([None, 4, 24 * 7, 16, 8, 1], [None, 1, 16, 8, 1], learning_rate=0.0002)
    filenames = os.listdir(config.dataset_path)
    files = [os.path.join(config.dataset_path, file) for file in filenames if file.endswith("NewEnd.h5")]
    dataset = STC_Provider(filenames=files[0], t_length=4, batch_size=16, input_size=24 * 7, output_size=1, splits=[3804, 348, 240])
    # dataset = STC_Provider(config.dataset_path + "/NYC14_M16x8_T60_NewEnd.h5", 7, 16, 24, 1, [3737, 415, 240])
    # dataset = STC_Provider(config.dataset_path + "/nyt_d_map.mat", 7, 16, 24, 1, [43824, 8760, 8760])
    model.fit(dataset, 100)
