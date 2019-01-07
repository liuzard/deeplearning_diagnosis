import scipy.io as scio
import os
import numpy as np

filepath="E:\GAN_test\\NORMAL"

'''
convert input vector to matrix：
para:vector-->input vertor to reshape
para:row-->number of rows of reshaped matrix
para:offset-->the begin of vector to reshape
'''
def file_reshape(vector,num_clounm,offset=0):
    num_row=110
    vector_use=vector[0:num_row*num_clounm,:]
    vector_reshape=vector_use.reshape(num_row,num_clounm)
    num_row_offset =0
    vector_use_offst = vector[offset:num_row_offset * num_clounm+offset, :]
    vector_reshape_offset=vector_use_offst.reshape(num_row_offset,num_clounm)
    vector_stack=np.row_stack((vector_reshape,vector_reshape_offset))
    return vector_stack

'''
read mat file and output the value according the given key
para:filepath-->filepath the matfile stores in
para:filename-->name of the matfile
para:key-->the key we want to read
'''
def matfile_reader(filepath,filename,key):
    file=filepath+'\\'+filename
    matfile=scio.loadmat(file)
    for k in matfile.keys():
        if key in k:
            return matfile[k]

listdir=os.listdir(filepath)
data_stack=np.zeros([0,1024])
for file in listdir:
    data=matfile_reader(filepath,file,'DE')
    data_reshape=file_reshape(data,1024,offset=0)
    data_stack=np.row_stack((data_stack,data_reshape))
scio.savemat(filepath[0:-7]+'\\'+filepath[-7:]+'.mat',{"data":data_stack})





