import numpy as np
import matplotlib.pyplot as plt
import re

#读取数据
def LoadData(filename):
    fo=open(filename)
    X=[]
    Y=[]
    for line in fo.readlines():
        line=re.split('\t| ',line.strip())
        X.append(list(map(float,line[:len(line)-1])))
        Y.append(int(line[-1]))
    return np.array(X),np.array(Y).reshape(len(Y),1)


#可视化数据
def ShowData(X,Y,graphname):
    plt.figure(graphname)
    for i in range(len(Y)):
        if(Y[i]==1):
            color='r'
        else:
            color='b'
        plt.scatter(X[i][0],X[i][1],c=color)
    plt.show()


#sigmoid函数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#代价函数
def costFunction(X,Y,theta):
    (m,n)=X.shape
    h=sigmoid(X.dot(theta))
    J=(-1/m)*((Y.T.dot(np.log(h)))+(1-Y).dot(np.log(1-h)))
    return J

#随机梯度下降
def gradient(X,Y,rate,maxloop):
    (m,n)=X.shape
    theta=np.ones((n,1))
    for i in range(maxloop):
        h=sigmoid(np.dot(X,theta))
        diff=h-Y
        pa_deri=(1/m)*np.dot(X.T,diff)
        theta=theta-rate*pa_deri
    return theta

(X,Y)=LoadData('linear.txt')
Xtr=X[:90,:]
Ytr=Y[:90]
Xte=X[90:,:]
Yte=Y[90:]
# ShowData(Xtr,Ytr,'data')
theta=gradient(Xtr,Ytr,0.01,1000)
print(theta)
print('\n')
print(sigmoid(np.dot(Xte,theta)))
print('\n')
print(Yte)