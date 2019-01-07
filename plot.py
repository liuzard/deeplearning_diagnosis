import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd

a=np.array(pd.read_excel('C:\\Users\liuzard\Desktop\研究\\诊断结果.xlsx'))
b=np.arange(1,11,1)
print(b)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.axis([0.5,10.5,99.6,100.02],fontsize=30)
plt.xlabel('实验序号',fontsize=40)
plt.ylabel('诊断精度（%）',fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
# plt.line
plt.plot(b,a[:,2],'r-o',label='测试精度',linewidth=3)
plt.plot(b,a[:,0],'b-D',label='训练精度',linewidth=3)
plt.legend(loc='best',fontsize=30)

plt.show()
