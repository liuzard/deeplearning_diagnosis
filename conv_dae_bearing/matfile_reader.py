import scipy.io as scio

def dataset_reader(path,train=True):
    data=scio.loadmat(path)
    if train == True:
        for k in data.keys():
            if 'train'in k:
                data=data[k][:,0:1024]
                return data
    else:
        for k in data.keys():
            if 'test' in k:
                data = data[k][:, 0:1024]
                return data