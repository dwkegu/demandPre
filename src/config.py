import os
import platform

CD_LT_LONG = 103.067980
CD_LT_LAT = 31.424122
CD_RB_LONG = 104.913659
CD_RB_LAT = 30.105347
CD_STARTTIME = 1477929600
CD_ENDTIME = 1480521600
CD_LONG_STEP = (CD_RB_LONG-CD_LT_LONG)/32
CD_LAT_STEP = (CD_LT_LAT - CD_RB_LAT)/32
CD_TIME_UNIT = 900

windows_path_prefix = 'f:/dataset/'
linux_path_prefix = '/home/pengshunfeng/f:/dataset/'
windows_project_path = "e:/PyCharm"
linux_project_path = "/home/pengshunfeng/pyCharmProject"

platform_string = platform.platform()
raw_dataset_path = windows_path_prefix if platform_string.__contains__("indow") else linux_path_prefix
project_path = windows_project_path if platform_string.__contains__("indow") else linux_project_path
dataset_path = project_path + "/demandPre/dataset"
log_path = project_path + "/demandPre/log"

cd_didi_order_path = raw_dataset_path + "didi/chengduDidi/order"
