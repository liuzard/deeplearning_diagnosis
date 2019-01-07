import scipy.io as scio
import numpy as np
import os

real_mat_data=scio.loadmat('C:\\Users\liuzard\PycharmProjects\deep_learning\CaseWestern_dataprocess\casewestern_bearing_Dataset.mat')
real_data=real_mat_data['train_dataset']
gener_mat_data=scio.loadmat('E:\generate\generate.mat')
gener_data=gener_mat_data['train_data']

fix_data=np.row_stack((real_data,gener_data))
np.random.shuffle(fix_data)

scio.savemat("fix_dataset.mat",{'train_dataset':fix_data,'test_dataset':real_mat_data['test_dataset']})