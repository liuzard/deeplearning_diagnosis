#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'a dft file'
__author__ = 'W.S.J.'
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio


bearing_path=r"G:\04 实验数据\02 实验台实验数据\实验数据_20180119\bearing_dataset_2000.mat"
bearing_data=scio.loadmat(bearing_path)["train_set"]
bearing_data=bearing_data[:,0:-1]
print(bearing_data.shape)

# from matplotlib.font_manager import fontManager
def plot_dft(x, y, z, Fs, N):
    #    plt.rcParams['font.sans-serif']=['SimHei']
    #    plt.rcParams['axes.unicode_minus']=False
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 18))
    line1, = axes[0].plot(x, 'b')  # the source signal
    axes[0].set_xlabel(u'sample dots')
    axes[0].set_ylabel(u'amplitude')
    axes[0].set_title(u'the source signal')
    aplittude= [abs(com_y/1000) for com_y in y[0:int(N / 2)]]
    a_array=np.array(aplittude)
    line2, = axes[1].plot([Fs / N * i for i in range(int(N / 2))], aplittude, 'r')
    axes[1].set_xlabel(u'frequency/Hz')
    axes[1].set_ylabel(u'amplitude')
    axes[1].set_title(u'the DFT signal amplitude')
    line3, = axes[2].plot([Fs / N * i for i in range(int(N / 2))],
                          [-math.atan(com_y.imag / com_y.real) if abs(com_y) > 0.1 else 0 for com_y in y[0:int(N / 2)]],
                          'r')
    axes[2].set_xlabel(u'frequency/Hz')
    axes[2].set_ylabel(u'phase')
    axes[2].set_title(u'the DFT signal phase')
    line4, = axes[3].plot(z, 'b')  # the source signal
    axes[3].set_xlabel(u'sample dots')
    axes[3].set_ylabel(u'amplitude')
    axes[3].set_title(u'the recovered signal by IDFT')
    fig.savefig('DFT_simulation.png', dpi=500, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    Fs = 20480  # sample rate
    N = 2000  # sample dots' number
    x=bearing_data[0]

    """
    f1 = 1000  # 1st component in signal
    x1 = [np.cos(2 * math.pi * f1 / Fs * n + math.pi / 8) for n in range(N)]
    f2 = 2000  # 2nd component in signal
    x2 = [np.cos(2 * math.pi * f2 / Fs * n - math.pi / 4) * 2 for n in range(N)]
    f3 = 3000  # 3rd component in signal
    x3 = [np.cos(2 * math.pi * f3 / Fs * n + math.pi / 2) * 3 for n in range(N)]
    x = [x1[i] + x2[i] + x3[i] for i in range(N)]  # the source signal
    """

    # contain the basis exp(-j*2*pi/N*k*n) and the projection weight
    y = []  # the result of DFT
    for k in range(N):
        basis = [complex(math.cos(2 * math.pi / N * k * n), math.sin(2 * math.pi / N * k * n)) for n in range(N)]
        y.append(np.dot(x, np.transpose(basis)))
    # contain the basis exp(j*2*pi/N*k*n) and the projection weight
    z = []  # the result of IDFT
    for k in range(N):
        basis = [complex(math.cos(2 * math.pi / N * k * n), -math.sin(2 * math.pi / N * k * n)) for n in range(N)]
        z.append(np.dot(y, np.transpose(basis)) / N)
    plot_dft(x, y, z, Fs, N)
    plt.show()
