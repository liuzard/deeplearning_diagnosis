import scipy.io as scio
import numpy as np
from AlexNet_1d_diagnosis import  matfile_reader

#parameters
noise_factor=0.5

#read data
data_path='E:/bearing_dataset_1024.mat'
train_dataset=matfile_reader.dataset_reader(data_path)
test_dataset=matfile_reader.dataset_reader(data_path,train=False)

train_data_sample=train_dataset[:,0:1024]
train_data_label=train_dataset[:,-1]

test_data_sample=test_dataset[:,0:1024]
test_data_label=test_dataset[:,-1]

#add noise to test datasamples and train datasamples
train_noise=train_data_sample+noise_factor*np.random.randn(*train_data_sample.shape)
test_noise=test_data_sample+noise_factor*np.random.randn(*test_data_sample.shape)

datasample_train_noise=np.column_stack((train_noise,train_data_label))
datasample_test_noise=np.column_stack((test_noise,test_data_label))

#save processed data
scio.savemat('E:/bearing_dataset_1024_with_noise.mat',{'traindata':datasample_train_noise,'testdata':datasample_test_noise})