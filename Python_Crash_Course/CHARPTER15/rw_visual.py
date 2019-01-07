import matplotlib.pyplot as plt
from Python_Crash_Course.CHARPTER15 import random_walk
while True:
    rw=random_walk.RandomWalk(5000)
    rw.fill_walk()
    point_numbers=list(range(rw.num_points))
    plt.plot(rw.x_values,rw.y_values,linewidth=2)
    # plt.scatter(rw.x_values[0],rw.y_values[0],c='green',s=20)
    # plt.scatter(rw.x_values[-1],rw.y_values[-1],c='red',s=20)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.show()
    plt.figure(figsize=(25,25))

    keep_runing=input("Make another walk?(y/n):")
    if keep_runing =='n':
        break