import scipy.io as scio
import os
import numpy as np

filepath="E:\西储大学数据集_程序用"

'''
convert input vector to matrix：
para:vector-->input vertor to reshape
para:row-->number of rows of reshaped matrix
para:offset-->the begin of vector to reshape
'''
def file_reshape(vector,num_clounm,offset=0):
    num_row=6
    vector_use=vector[offset:num_row*num_clounm+offset,:]
    vector_reshape=vector_use.reshape(num_row,num_clounm)
    return vector_reshape

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

def datasetconstruct(data,label):
    num_row=data.shape[0]
    data_label=label*np.ones([num_row,1])
    dataset=np.column_stack((data,data_label))
    return dataset

listdir=os.listdir(filepath)
train_data_stack=np.zeros([0,1025])
test_dataset_stack=np.zeros([0,1025])
offset_data_stack=np.zeros([0,1025])
for file in listdir:
    data=matfile_reader(filepath,file,'DE')
    data_reshape=file_reshape(data,1024,offset=512)
    if 'normal' in file:
        label=0
        dataset=datasetconstruct(data_reshape,label)
    if 'B007' in file:
        label=1
        dataset=datasetconstruct(data_reshape,label)
    if 'B014' in file:
        label=2
        dataset=datasetconstruct(data_reshape,label)
    if 'B021' in file:
        label=3
        dataset=datasetconstruct(data_reshape,label)
    if 'B028' in file:
        label=4
        dataset=datasetconstruct(data_reshape,label)
    if 'IR007' in file:
        label=5
        dataset=datasetconstruct(data_reshape,label)
    if 'IR014' in file:
        label=6
        dataset=datasetconstruct(data_reshape,label)
    if 'IR021' in file:
        label=7
        dataset=datasetconstruct(data_reshape,label)
    if 'IR028' in file:
        label=8
        dataset=datasetconstruct(data_reshape,label)
    if 'OR007' in file:
        label=9
        dataset=datasetconstruct(data_reshape,label)
    if 'OR014' in file:
        label=10
        dataset=datasetconstruct(data_reshape,label)
    if 'OR021' in file:
        label=11
    dataset=datasetconstruct(data_reshape,label)
    offset_data_stack=np.row_stack((offset_data_stack,dataset))
   #no offset
    # num_train=6
    # train_dataset=dataset[0:7,:]
    # test_dataset=dataset[7:,:]
    # train_data_stack=np.row_stack((train_data_stack,train_dataset))
    # test_dataset_stack=np.row_stack((test_dataset_stack,test_dataset))

scio.savemat("casewestern_bearing_Dataset_offset.mat",{"train_dataset":offset_data_stack})





