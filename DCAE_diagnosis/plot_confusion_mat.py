import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(font='SimHei', style="white")  # 解决Seaborn中文显示问题
import pandas as pd

excel_path=r"C:\Users\liuzard\PycharmProjects\deep_learning_diagnosis\DCAE_diagnosis\confusion_bearing.xlsx"
conf_mat=pd.read_excel(excel_path)
cm=np.array(conf_mat)
print(cm)
labels=range(10)


tick_marks = np.array(range(10)) + 0.5

def plot_confusion_matrix(cm, title=None, cmap=plt.get_cmap('gray_r')):
    plt.imshow(cm, interpolation='nearest',cmap="gray_r")
    # plt.title(title)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(0, 100, 6))
    cbar.set_ticklabels(('0', '20', '40', '60', '80', '100',"120"))
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, fontsize=15)
    plt.yticks(xlocations, labels, fontsize=15)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predict label', fontsize=20)


np.set_printoptions(precision=5)
cm_normalized = cm
plt.figure(figsize=(12, 8))

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    # if c > 0.01:
    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=13, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm)
plt.show()
