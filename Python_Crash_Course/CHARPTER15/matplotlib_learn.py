import matplotlib.pyplot as plt

#绘制平方数折折线
x=[1,2,3,4,5]
squares=[1,4,9,16,25]
plt.plot(x,squares,linewidth=5)

#设置图表标题，横纵坐标名称
plt.title("Square Number",fontsize=24)
plt.xlabel("value",fontsize=14)
plt.ylabel("square of value",fontsize=14)

#设置刻度标记的大小
plt.tick_params(axis="both",labelsize=14)


plt.show()
