
#给定一个数据集与标签，标签为（1，-1），训练二分类
import numpy as np

class Classifier:
    def __init__(self):
        self.label=1;
        self.feature_index=None#划分特征
        self.door=None#划分边界的特征值
        self.weights=None#分类器权重

def AdaBoost(X,y,n_of_classifiers):
    m=X.shape[0]
    n=X.shape[1]
    w=np.full(m,1/m)#训练数据的权重
    classifiers=[]#分类器列表
    for r in range(n_of_classifiers):#对于每一个分类器
        classifier=Classifier()
        min_error=99999
        for i in range(n):#对于X的每一个特征
            unique_character_values=np.unique(X[:,i])#特征的取值列表
            for door in unique_character_values:#door为分割阈值
                p=1#p=1意味着特征值大于阈值判断为1；p=-1意味着特征值小于阈值判断为-1
                predict=np.ones(np.shape(y))
                predict[X[:,i]<door]=-1
                error=np.sum(w[y!=predict])
                if(error>0.5):
                    error=1-error
                    p=-1
                if error<min_error:
                    min_error=error
                    classifier.door=door
                    classifier.feature_index=i
                    classifier.label=p
        classifier.weights=0.5*np.log((1.0-min_error)/(min_error+1e-9))
        predict=np.ones(np.shape(y))
        index=(classifier.label*X[:,classifier.feature_index]<classifier.label*classifier.door)
        predict[index]=-1
        w=w*np.exp(-classifier.weights*y*predict)
        w=w/np.sum(p)
        classifiers.append(classifier)
    return classifiers

def AdaBoostPredict(classifiers,X):
    predict=np.zeros((X.shape[0],1))
    for classifier in classifiers:
        predict_i=np.ones((X.shape[0],1))
        index=(classifier.label*X[:,classifier.feature_index]<classifier.label*classifier.door)
        predict_i[index]=-1
        predict=predict+classifier.weights*predict_i
    predict[predict<0]=-1
    predict[predict>0]=1
    return predict


from sklearn.model_selection import  train_test_split
from sklearn.datasets._samples_generator import make_blobs
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
X,y=make_blobs(n_samples=150,n_features=2,centers=2,cluster_std=1.2,random_state=40)
y_=y.copy()
y_[y<=0]=-1
y_[y>0]=1
y_=y_.astype(float)
# 训练/测试数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_,
 test_size=0.3, random_state=43)
# 设置颜色参数
colors = {0:'r', 1:'g'}
# 绘制二分类数据集的散点图
plt.scatter(X[:,0], X[:,1], marker='o', c=pd.Series(y).map(colors))
plt.show();

classifiers=AdaBoost(X_train,y_train,5)
predict=AdaBoostPredict(classifiers,X_test)
accuracy = accuracy_score(y_test, predict)
print("Accuracy of AdaBoost by sklearn:", accuracy)