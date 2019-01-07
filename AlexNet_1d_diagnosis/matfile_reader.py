import scipy.io as scio


def dataset_reader(path, train=True):
    data = scio.loadmat(path)
    if train:
        for k in data.keys():
            if 'train' in k:
                return data[k]
    else:
        for k in data.keys():
            if 'test' in k:
                return data[k]
