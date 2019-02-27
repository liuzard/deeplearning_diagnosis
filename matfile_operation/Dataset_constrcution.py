import scipy.io as scio
import numpy as np
import os

bearing_file_path = r"G:\04 实验数据\02 实验台实验数据\实验数据_20180119\轴承"
gear_file_path = r"G:\04 实验数据\02 实验台实验数据\实验数据_20180119\齿轮箱"

bearing_label_list = ["normal", "inner_01", "inner_02", "inner_03", "outer_01", "outer_02", "outer_03", "roller_01",
                      "roller_02", "roller_03"]
gear_label_list = ["normal", "liewen", "diansi", "toothmissing", "tear"]
dataset_savepath = r"G:\04 实验数据\02 实验台实验数据\实验数据_20180119"


# 获取所有文件的路径，以列表的形式返回
def get_file_list(matfile_path):
    """
    :param matfile_path: 多个原始文件所在文件夹的路径
    :return: 所有文件的路径，以列表的形式
    """
    dir_list = os.listdir(matfile_path)
    file_path_list = []
    for file in dir_list:
        file_path = os.path.join(matfile_path, file)
        file_path_list.append(file_path)
    return file_path_list


# 根据文件名获取该文件存储数据的故障类型，用数字来表示，1,2,3...
def get_label(matfile, label_list):
    """
    :param matfile: mat 文件的路径
    :param label_list: 故障代表的label,根据故障在列表中的顺序确定其label 数字
    :return: 文件中数据的label
    """
    lable = None
    for i in range(len(label_list)):
        if label_list[i] in matfile:
            label = i
            return label


def dataset_construction(matfile_path, label_list, sample_length=2000, trainset_num=2000, testset_num=500):
    file_list = get_file_list(matfile_path)
    train_set = np.zeros((0, sample_length + 1))
    test_set = np.zeros((0, sample_length + 1))
    for file in file_list:
        label = get_label(file, label_list)
        data_point = sample_length * (trainset_num + testset_num)
        data = scio.loadmat(file)["data1"][0:data_point, 1]
        data_reshape = data.reshape(-1, sample_length)
        label_column = label * np.ones((data_reshape.shape[0], 1))
        samples = np.column_stack((data_reshape, label_column))
        np.random.shuffle(samples)
        train_part = samples[0:trainset_num]
        test_part = samples[trainset_num:trainset_num + testset_num]
        train_set = np.row_stack((train_set, train_part))
        test_set = np.row_stack((test_set, test_part))
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)
    return train_set, test_set


def save_dataset(save_path, filename, train_set, test_set):
    full_path = os.path.join(save_path, filename)
    scio.savemat(full_path, {"train_set": train_set, "test_set": test_set})


# 轴承样本集生成
bearing_train_set, bearing_test_set = dataset_construction(bearing_file_path, bearing_label_list)
save_dataset(dataset_savepath, "bearing_dataset_2000.mat", bearing_train_set, bearing_test_set)

# 齿轮箱样本集生成
# gear_train_set,gear_test_set=dataset_construction(gear_file_path,gear_label_list)
# save_dataset(dataset_savepath,"gear_dataset_2000.mat",gear_train_set,gear_test_set)
