import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
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

#多分类梯度下降
def regrssion_oneVSall(all_class,X,Y,rate,maxloop):
    # thetas=np.empty((X.shape[1],len(all_class)))
    thetas=np.zeros((X.shape[1],1))
    for index,cla in enumerate((all_class)):
        newY=np.zeros(Y.shape)
        newY[np.where(Y==cla)]=1
        theta_i=gradient(X,newY,rate,maxloop)
        # print(gradient(X,newY,rate,maxloop).shape)
        thetas=np.concatenate((thetas,theta_i),axis=1)
        # np.concatenate((thetas,gradient(X,newY,rate,maxloop)),axis=1)
    return thetas[:,1:]

#预测正确率
def predict(Xte,Yte,thetas,all_class):
    Y_pred=np.empty(Yte.shape)
    for index,row in enumerate(np.dot(Xte,thetas)):#enumerate函数返回标号和元素
        Y_pred[index]=all_class[np.argmax(row)]
    Y_pred=Y_pred.astype(np.int32)
    pred_rate= np.array(Yte==Y_pred)
    return np.mean(pred_rate.reshape(Yte.shape))

# def splitData(X,Y):
#     spli_num=int(len(Y)*0.90)
#     Xtr=X[:spli_num,:]
#     Ytr=Y[:spli_num]
#     Xte=X[spli_num:,:]
#     Yte=Y[spli_num:]
#     return Xtr,Ytr,Xte,Yte

fo=loadmat('ex3data1.mat')
X=fo['X']
Y=fo['y']
Xtr=X
Ytr=Y
Xte=X[range(5000)[49::50]]
Yte=Y[range(5000)[49::50]]

all_class=list(set(np.ravel(Ytr)))
thetas=regrssion_oneVSall(all_class,Xtr,Ytr,0.01,1000)
print(predict(Xte,Yte,thetas,all_class))
