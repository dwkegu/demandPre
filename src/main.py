import sys
import platform
import tensorflow as tf

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
    model = STC_Lstm([None, 10, 12, 64, 64, 1], [None, 1, 64, 64, 1])
    dataset = STC_Provider(config.dataset_path + "/d_map.mat", 10, 8, 12, 1, 0.7)
    model.fit(dataset, 100)
