import scipy.io as scio
import numpy as np
import os

mat_file=scio.loadmat('casewestern_bearing_Dataset.mat')
mat_file_offset=scio.loadmat('casewestern_bearing_Dataset_offset.mat')

data=mat_file['train_dataset']
data_offset=mat_file_offset['train_dataset']

train_data=np.row_stack((data,data_offset))
np.random.shuffle(train_data)

scio.savemat('fix_dataset.mat',{'train_dataset':train_data,'test_dataset':mat_file['test_dataset']})
