import  scipy.io as scio
import pandas as pd
from AlexNet_1d_diagnosis  import matfile_reader
import numpy as np



train_data=matfile_reader.dataset_reader("E:\\bearing_dataset_2000.mat")
test_data=matfile_reader.dataset_reader('E:\\bearing_dataset_2000.mat',train=False)

np.savetxt('train_data.csv', train_data,delimiter = ',')
np.savetxt('test_data.csv', test_data,delimiter = ',')

