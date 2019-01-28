import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np

roller_mat_path = r"C:\Users\liuzard\Desktop\factory1.mat"
roller_mat = scio.loadmat(roller_mat_path)
# print(roller_mat)
roller_data = roller_mat["factory1"]
print(roller_data.shape)
original_signal = roller_data[6000:8000, 0]


def add_noise(original_signal, noise_factor):
    Gaussian_noise = noise_factor * np.random.randn(*original_signal.shape)
    noisy_signal = Gaussian_noise + original_signal
    return noisy_signal


def plot(data, linewidth, ylabel, ymin=-10000, ymax=10000):
    # plt.rcParams['xtick.labelsize'] = 30
    # plt.rcParams['ytick.labelsize'] = 30
    plt.plot(data, linewidth=linewidth)
    plt.xlim(0, 2010)
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel, fontsize=15)


noisy_signal = add_noise(original_signal, 0.35)
reconstruct_signal_dae = add_noise(original_signal, 0.16)
reconstruct_signal_dcae = add_noise(original_signal, 0.08)


def plot_signals(orignal_data, noisy_data, re_data_dcae, re_data_dae):
    # sn.set(style="whitegrid", font='SimHei', color_codes=True)
    # plt.subplot(311)
    # plot(train_data, 1, '干净信号')

    plt.subplot(221)
    plot(orignal_data, 1, 'Original signal')

    plt.subplot(222)
    plot(noisy_data, 1, 'Noisy signal')

    plt.subplot(224)
    plot(re_data_dcae, 1, 'De-noised signal (DAE)')

    plt.subplot(223)
    plot(re_data_dae, 1, 'De-noised signal (DCAE)')


plot_signals(original_signal, noisy_signal, reconstruct_signal_dae, reconstruct_signal_dcae)
plt.show()
