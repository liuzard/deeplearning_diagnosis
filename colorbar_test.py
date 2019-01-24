import os
import numpy as np
import wlab  # pip install wlab
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import griddata

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
# **********************************************************
FreqPLUS = ['F06925', 'F10650', 'F23800', 'F18700', 'F36500', 'F89000']
#
FindPath = '/d3/MWRT/R20130805/'
# **********************************************************
fig = plt.figure(figsize=(8, 6), dpi=72, facecolor="white")
axes = plt.subplot(111)
axes.cla()  # 清空坐标轴内的所有内容
# 指定图形的字体
font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }
# **********************************************************
# 查找目录总文件名中保护F06925，EMS和txt字符的文件
for fp in FreqPLUS:
    FlagStr = [fp, 'EMS', 'txt']
    FileList = wlab.GetFileList(FindPath, FlagStr)
    #
    LST = []  # 地表温度
    EMS = []  # 地表发射率
    TBH = []  # 水平极化亮温
    TBV = []  # 垂直极化亮温
    #
    findex = 0
    for fn in FileList:
        findex = findex + 1
        if (os.path.isfile(fn)):
            print(str(findex) + '-->' + fn)
            # fn='/d3/MWRT/R20130805/F06925_EMS60.txt'
            data = wlab.dlmread(fn)
            EMS = EMS + list(data[:, 1])  # 地表发射率
            LST = LST + list(data[:, 2])  # 温度
            TBH = TBH + list(data[:, 8])  # 水平亮温
            TBV = TBV + list(data[:, 9])  # 垂直亮温
    # -----------------------------------------------------------
    # 生成格点数据，利用griddata插值
    grid_x, grid_y = np.mgrid[275:315:1, 0.60:0.95:0.01]
    grid_z = griddata((LST, EMS), TBH, (grid_x, grid_y), method='cubic')
    # 将横纵坐标都映射到（0，1）的范围内
    extent = (0, 1, 0, 1)
    # 指定colormap
    cmap = matplotlib.cm.jet
    # 设定每个图的colormap和colorbar所表示范围是一样的，即归一化
    norm = matplotlib.colors.Normalize(vmin=160, vmax=300)
    # 显示图形，此处没有使用contourf #>>>ctf=plt.contourf(grid_x,grid_y,grid_z)
    gci = plt.imshow(grid_z.T, extent=extent, origin='lower', cmap=cmap, norm=norm)
    # 配置一下坐标刻度等
    ax = plt.gca()
    ax.set_xticks(np.linspace(0, 1, 9))
    ax.set_xticklabels(('275', '280', '285', '290', '295', '300', '305', '310', '315'))
    ax.set_yticks(np.linspace(0, 1, 8))
    ax.set_yticklabels(('0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95'))
    # 显示colorbar
    cbar = plt.colorbar(gci)
    cbar.set_label('$T_B(K)$', fontdict=font)
    cbar.set_ticks(np.linspace(160, 300, 8))
    cbar.set_ticklabels(('160', '180', '200', '220', '240', '260', '280', '300'))
    # 设置label
    ax.set_ylabel('Land Surface Emissivity', fontdict=font)
    ax.set_xlabel('Land Surface Temperature(K)', fontdict=font)  # 陆地地表温度LST
    # 设置title
    titleStr = '$T_B$ for Freq = ' + str(float(fp[1:-1]) * 0.01) + 'GHz'
    plt.title(titleStr)
    figname = fp + '.png'
    plt.savefig(figname)
    plt.clf()  # 清除图形

# plt.show()
print('ALL -> Finished OK')