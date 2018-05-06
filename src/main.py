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


if __name__ == '__main__':
    model = CubeCNN([None, 24*7, 64, 64, 1], [None, 1, 64, 64, 1])
    dataset = DidiDataProvider(config.dataset_path + "/demand_map.npy", 8, 24*7, 1)
    model.fit(dataset, 100)