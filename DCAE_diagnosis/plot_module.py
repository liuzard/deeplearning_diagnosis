import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import tensorflow as tf
import pandas as pd
from pylab import mpl
from matplotlib.font_manager import FontProperties
import seaborn as sns

from matplotlib.font_manager import FontProperties

myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
sns.set(font=myfont.get_name())
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei', style="white")  # 解决Seaborn中文显示问题
# sns.set(style="white")     #set( )设置主题，调色板更常用


file_path = r"F:\User\LiuXingChen\故障诊断实验数据\实验数据_20180119\轴承\mat\preprocess_2000\04roller_01_ch01_h_ch02_v.mat"
data_excel_path = r'C:\Users\liuzard\PycharmProjects\deep_learning_diagnosis\DCAE_diagnosis\experiment1.xlsx'



def dataset_reader(path, train=True):
    data = scio.loadmat(path)
    if train == True:
        for k in data.keys():
            if 'data' in k:
                return data[k]
    else:
        for k in data.keys():
            if 'test' in k:
                return data[k]


def reverse(input_array):
    data_list = []
    for i in range(input_array.shape[0]):
        data_list.append(input_array[input_array.shape[0] - i - 1])
    data_list = np.array(data_list)
    return data_list


myfont = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=40)

mpl.rcParams['font.sans-serif'] = ['simsun']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.pad'] = 10
mpl.rcParams['ytick.minor.pad'] = 10
plt.rcParams['xtick.minor.size'] = 10
plt.rcParams['ytick.minor.size'] = 10


def plot(data, linewidth, ylabel, ymin=-3, ymax=3):
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.plot(data, linewidth=linewidth)
    plt.xlim(0, 2010)
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel, fontsize=30)


def plot_signals(train_data, noise, noise_signal):
    # sn.set(style="whitegrid", font='SimHei', color_codes=True)
    # plt.subplot(311)
    # plot(train_data, 1, '干净信号')

    plt.subplot(312)
    plot(noise, 1, '高斯噪声')

    plt.subplot(313)
    plot(noise_signal, 1, '含噪信号')

    plt.subplot(311)
    plot(train_data, 1, '干净信号')


def plot_multomodel_comparasion(ACNN_acc, SDAE_acc,SDAE_denoise_acc,WDCNN_acc,WDCNN_denoise_acc,bp_acc, svm_acc):
    x = range(-2, 13, 1)
    plt.figure(figsize=(16, 7))
    plt.plot(x, ACNN_acc, 'b-D', label='DCAE-1D + ACNNDM-1D (proposed)', markersize=7, linewidth=2)
    plt.plot(x, SDAE_acc, 'r-s', label='SDAE', markersize=7, linewidth=2)
    plt.plot(x, SDAE_denoise_acc, 'r->', label='DCAE-1D + SDAE', markersize=7, linewidth=2)
    plt.plot(x, WDCNN_acc, 'g-<', label='WDCNN', markersize=7, linewidth=2)
    plt.plot(x, WDCNN_denoise_acc, 'k-*', label='DCAE-1D + WDCNN', markersize=7, linewidth=2)
    plt.plot(x, bp_acc, 'g-o', label='DCAE-1D + BP', markersize=7, linewidth=2)
    plt.plot(x, svm_acc, 'c-X', label='DCAE-1D + SVM', markersize=7, linewidth=2)
    # plt.gca().spines['top'].set_visible(False)  # 去掉上边框
    # plt.gca().spines['right'].set_visible(False)  # 去掉右边框
    plt.tick_params(length=10)
    plt.ylabel("Diagnosis accuracy(%)", fontsize=25)
    plt.xlabel("SNR(dB)", fontsize=25)

    plt.legend(fontsize=20, loc='lower right')
    plt.xticks(np.arange(-2, 13, 1), fontsize=25)
    plt.yticks(np.arange(50, 101, 10), fontsize=25)
    plt.grid(axis='y',b=True)
    plt.savefig(r"C:\Users\liuzard\Desktop\figs\multi_models_compare.png", format='png', transparent=True,
                dpi=400)


def plot_reconstruct_error(dcae_error, dae_error):
    x = range(-2, 13, 1)
    plt.figure(figsize=(16, 7))
    plt.plot(x, dcae_error, 'b-D', label='DCAE-1D', markersize=8, linewidth=2)
    plt.plot(x, dae_error, 'r-s', label='DAE', markersize=8, linewidth=2)
    # plt.gca().spines['top'].set_visible(False)  # 去掉上边框
    # plt.gca().spines['right'].set_visible(False)  # 去掉右边框
    plt.tick_params(length=10)
    plt.ylabel("Reconstruct error", fontsize=30)
    plt.xlabel("SNR(dB)", fontsize=30)
    # plt.xlim(-2.1, 12.1)
    # plt.ylim(0, 0.06)
    plt.legend(fontsize=30, loc='upper right')
    plt.xticks(np.arange(-2, 13, 1), fontsize=30)
    plt.yticks(np.arange(0, 0.025, 0.005), fontsize=30)
    plt.grid(axis='y')
    plt.savefig(r"C:\Users\liuzard\Desktop\figs\recons_gear.png", format='png', transparent=True,
                dpi=400)


def plot_dcae_dae_none_compara(dcae_acc, dae_acc, non_acc):
    x = range(-2, 13, 1)
    plt.figure(figsize=(16, 7))
    plt.plot(x, dcae_acc, 'b-D', label='DCAE-1D + AICNN-1D', markersize=7, linewidth=2)
    plt.plot(x, dae_acc, 'r-s', label='DAE + AICNN-1D', markersize=7, linewidth=2)
    plt.plot(x, non_acc, 'g-o', label='AICNN-1D without denoise', markersize=7, linewidth=2)
    # plt.plot(x, svm_acc, 'y-X', label='SVM', markersize=8, linewidth=2)
    # plt.gca().spines['top'].set_visible(False)  # 去掉上边框
    # plt.gca().spines['right'].set_visible(False)  # 去掉右边框
    # plt.gca().spines['left'].set_visible(False)  # 去掉上边框
    # plt.gca().spines['bottom'].set_visible(False)  # 去掉右边框
    plt.tick_params(length=10)
    plt.ylabel("Diagnosis Accuracy(%)", fontsize=30)
    plt.xlabel("SNR(dB)", fontsize=30)

    plt.legend(fontsize=30, loc='lower right')
    plt.xticks(np.arange(-2, 13, 1), fontsize=30)
    plt.yticks(np.arange(90, 100.1, 2), fontsize=30)
    plt.grid(axis='y')
    plt.savefig(r"C:\Users\liuzard\Desktop\figs\compare_diff_noise_bearing.png", format='png', transparent=True, dpi=400)

def plot_drop_pooling_val(full_acc,without_corrupt_acc,without_pooling_acc):
    x = range(-2, 13, 1)
    plt.figure(figsize=(16, 7))
    plt.plot(x, full_acc, 'b-D', label='DCAE-1D + AICNN-1D (model A)', markersize=7, linewidth=2)
    plt.plot(x, without_corrupt_acc, 'r-s', label='DCAE-1D + AICNN-1D (model B)', markersize=7, linewidth=2)
    plt.plot(x, without_pooling_acc, 'g-o', label='DCAE-1D + AICNN-1D (model C)', markersize=7, linewidth=2)
    # plt.plot(x, svm_acc, 'y-X', label='SVM', markersize=8, linewidth=2)
    # plt.gca().spines['top'].set_visible(False)  # 去掉上边框
    # plt.gca().spines['right'].set_visible(False)  # 去掉右边框
    plt.tick_params(length=10)
    plt.ylabel("Diagnosis Accuracy(%)", fontsize=30)
    plt.xlabel("SNR(dB)", fontsize=30)
    # plt.xlim(-2.1, 12.1)
    # plt.ylim(90, 100.1)
    plt.legend(fontsize=30, loc='lower right')
    plt.xticks(np.arange(-2, 13, 1), fontsize=30)
    plt.yticks(np.arange(90, 100.1, 2), fontsize=30)
    plt.grid(axis='y')
    plt.savefig(r"C:\Users\liuzard\Desktop\figs\drop_pool_val.png", format='png', transparent=True,
                dpi=400)



def plot_single_curve(ACNN_acc):
    x = reverse(np.array(range(-2, 13, 1)))
    plt.plot(x, ACNN_acc, 'b-D', markersize=8, linewidth=2)
    # plt.gca().spines['top'].set_visible(False)  # 去掉上边框
    # plt.gca().spines['right'].set_visible(False)  # 去掉右边框
    plt.tick_params(length=10)
    plt.ylabel("Diagnosis Accuracy(%)", fontsize=30)
    plt.xlabel("Signal-Noise Ratio(dB)", fontsize=30)
    # plt.legend(fontsize=30, loc='lower right')
    plt.xticks(np.arange(12, -2.1, -1), fontsize=30)
    plt.yticks(np.arange(40, 101, 10), fontsize=30)
    plt.grid(axis='both')


def plot_nondrop_comparasion(drop_acc, nondrop_acc):
    x = reverse(np.array(range(-2, 13, 1)))
    plt.plot(x, drop_acc, 'b->', label='model with input dropout', markersize=8, linewidth=2)
    plt.plot(x, nondrop_acc, 'r-s', label='model without input dropout', markersize=8, linewidth=2)
    plt.ylabel("Diagnosis accuracy(%)", fontproperties=myfont, fontsize=30)
    plt.xlabel("SNR(dB)", fontproperties=myfont, fontsize=30)
    plt.xlim(12.2, -1.8)
    plt.ylim(40, 102)
    plt.legend(fontsize=30, loc='lower left')
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    # plt.gca().spines['top'].set_visible(False)  # 去掉上边框
    # plt.gca().spines['right'].set_visible(False)  # 去掉右边框
    plt.tick_params(length=10)
    plt.xticks(np.arange(12, -2.1, -1), fontproperties=myfont)
    plt.yticks(np.arange(40, 101, 10), fontproperties=myfont)
    plt.grid(axis='both')


def plot_fc_avg_compara(avg_acc, fc_acc):
    # sn.set(style="white", font='SimHei', palette="muted", color_codes=True)
    x = reverse(np.array(range(-2, 13, 1)))
    plt.plot(x, avg_acc, 'b->', label='ACNNDM-1D（全局平均）', linewidth=2, markersize=8)
    plt.plot(x, fc_acc, 'r-s', label='ACNNDM-1D（全连接）', linewidth=2, markersize=8)
    plt.ylabel("诊断精度(%)", fontsize=30)
    plt.xlabel("信噪比(dB)", fontsize=30)
    plt.xlim(-2.1, 12.1)
    plt.ylim(75, 102)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.gca().spines['top'].set_visible(False)  # 去掉上边框
    plt.gca().spines['right'].set_visible(False)  # 去掉右边框
    plt.tick_params(length=10)
    plt.legend(fontsize=30, loc='lower right')
    plt.xticks(np.arange(12, -2.1, -1), fontsize=30)
    plt.yticks(np.arange(70, 101, 10), fontsize=30)
    # plt.grid(axis='both')


def plot_acc_diffent_dropout(data_drop_0, data_drop_2, data_drop_4, data_drop_6, data_drop_8, data_drop_0_8):
    # sn.set(style="white", font='SimHei', palette="muted", color_codes=True)
    x = reverse(np.array(range(-2, 13, 1)))
    plt.plot(x, data_drop_0, 'g-D', label='破坏率 = 0', linewidth=2, markersize=8)
    plt.plot(x, data_drop_2, 'b-s', label='破坏率 = 0.2', linewidth=2, markersize=8)
    plt.plot(x, data_drop_4, color="greenyellow", linestyle="-.", marker="X", label='破坏率 = 0.4', linewidth=2,
             markersize=8)
    plt.plot(x, data_drop_6, color="grey", linestyle="--", marker="o", label='破坏率 = 0.6', linewidth=2, markersize=8)
    plt.plot(x, data_drop_8, 'r-*', label='破坏率 = 0.8', linewidth=2, markersize=10)
    plt.plot(x, data_drop_0_8, 'k->', label='破坏率 = 0～0.8', linewidth=2, markersize=8)
    plt.ylabel("诊断精度(%)", fontsize=30)
    plt.xlabel("信噪比(dB)", fontsize=30)
    plt.gca().spines['top'].set_visible(False)  # 去掉上边框
    plt.gca().spines['right'].set_visible(False)  # 去掉右边框
    plt.tick_params(length=10)
    plt.xlim(-2.1, 12.1)
    plt.ylim(75, 102)
    plt.legend(fontsize=30, loc='lower right')
    plt.xticks(np.arange(12, -2.1, -1), fontsize=30)
    plt.yticks(np.arange(0, 0.05, 0.01), fontsize=30)
    # plt.grid(axis='both')


data_testbed = np.array(pd.read_excel(data_excel_path))

# 轴承数据集采用不同的自编码器的重构误差
bearing_dcae_error = data_testbed[:, 2]
bearing_dae_error = data_testbed[:, 6]

# 齿轮箱数据集采用不同的自编码器的重构误差
gear_dcae_error = data_testbed[:, 27]
gear_dae_error = data_testbed[:, 28]

# 轴承数据集在卷积自编码器去噪、自编码器去噪和无去噪三种条件下的诊断精度
bearing_dcae_acc = data_testbed[:, 7]
bearing_dae_acc = data_testbed[:, 8]
bearing_none_acc = data_testbed[:, 9]

# 齿轮数据集在卷积自编码器去噪、自编码器去噪和无去噪三种条件下的诊断精度
gear_dcae_acc = data_testbed[:, 29]
gear_dae_acc = data_testbed[:, 30]
gear_none_acc = data_testbed[:, 31]

# 轴承数据集在完整条件下，干净破坏输入条件下和全局平均池化条件下的诊断精度对比
bearing_full_acc=data_testbed[:,7]
bearing_clean_train_acc=data_testbed[:,10]
bearing_fc_acc=data_testbed[:,11]

# 不同模型诊断精度对比 AICNN-1D,LeNet-2D,BP,SVM
AICNN_acc=data_testbed[:,7]
SDAE_acc= data_testbed[:,13]
SDAE_denoise_acc= data_testbed[:,14]
WDCNN_acc=data_testbed[:,15]
WDCNN_denoise_acc=data_testbed[:,16]
bp_acc=data_testbed[:,17]
svm_acc=data_testbed[:,18]


# 重构误差曲线图
# plot_reconstruct_error(bearing_dcae_error,bearing_dae_error)
# plot_reconstruct_error(gear_dcae_error,gear_dae_error)

# 不同去噪策略下的故障诊断曲线图
# plot_dcae_dae_none_compara(bearing_dcae_acc,bearing_dae_acc,bearing_none_acc)
plot_dcae_dae_none_compara(gear_dcae_acc,gear_dae_acc,gear_none_acc)

# 不同训练方式下的故障诊断曲线图
# plot_drop_pooling_val(bearing_full_acc,bearing_clean_train_acc,bearing_fc_acc)

# 不同模型的对比曲线图
# plot_multomodel_comparasion(AICNN_acc, SDAE_acc, SDAE_denoise_acc,WDCNN_acc,WDCNN_denoise_acc,bp_acc,svm_acc)



plt.show()
