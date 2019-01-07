import matplotlib.pyplot as plt

x_values=list(range(1,1001))
y_values=[x*x*x for x in x_values]

plt.scatter(x_values,y_values,c=y_values,cmap=plt.cm.Greens,s=10)
plt.title("square scatter",fontsize=20)
plt.xlabel("value",fontsize=14)
plt.ylabel("square of value",fontsize=14)
plt.tick_params(axis="both",which='major',labelsize=14)

# plt.axis([0,1100,0,1100000])
plt.savefig('scatter_square.png',bboxs_inches='tight')
plt.show()