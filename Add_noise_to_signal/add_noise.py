import math
import scipy.io as scio
import numpy as np

signal_path=r"G:\04 实验数据\bearing_dataset_2000.mat"
signal=scio.loadmat(signal_path)["train_dataset"][0][0:2000]

