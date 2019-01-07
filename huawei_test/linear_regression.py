import matplotlib.pyplot as plt


X=[1,2,3,4,5,6,7]
Y=[3.09,5.06,7.5,9.12,10.96,6.91,19.01]

def linear_regression(X,Y):
    Xsum=0.0
    X2sum=0.0
    Ysum=0.0
    XY=0.0
    n=len(X)
    for i in range(n):
        Xsum += X[i]
        Ysum+=Y[i]
        XY+=X[i]*Y[i]
        X2sum+=X[i]**2
    k=(Xsum*Ysum/n-XY)/(Xsum**2/n-X2sum)
    b=(Ysum-k*Xsum)/n
    return k,b

k,b = linear_regression(X,Y)
print('the line is y=%f*x+%f' % (k,b) )
x_fit=range(1,9)
y_fit=[k*x+b for x in x_fit]
for i in x_fit:
    print(i)
print(x_fit,y_fit)
plt.scatter(X,Y)
plt.plot(x_fit,y_fit)


plt.show()