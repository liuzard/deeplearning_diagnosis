import scipy.io as scio
import numpy as np
import os


file_path="E:\generate"
list_dir=os.listdir(file_path)
print(list_dir)
data_stack=np.zeros([0,1025])
for file in list_dir:
    file_name=file_path+'\\'+file
    mat_data=scio.loadmat(file_name)
    data = mat_data['data']

    if 'NORMAL' in file:
        label=0
    if 'B007' in file:
        label=1
    if 'B014' in file:
        label=2
    if 'B021' in file:
        label=3
    if 'B028' in file:
        label=4
    label_column = label*np.ones([data.shape[0], 1])
    data=np.column_stack((data,label_column))
    data_stack=np.row_stack((data_stack,data))
    np.random.shuffle(data_stack)
scio.savemat("E:\generate\\generate.mat",{"train_data":data_stack})





