
import numpy as np
import pandas as pd
from math import log

def gini(X):#计算基尼指数
    probabilityS=[]
    gini=0
    X_=[i for i in X]
    X.reshape((1,-1))
    for i in np.unique(X):
        probability=X_.count(i)/len(X_)
        probabilityS.append(probability)
    for i in probabilityS:
        gini+=i*(1-i)
    return gini

def feature_split(Xy, feature_i, door):#根据特征的取值分裂数据集分裂
    unique_value = np.unique(Xy[:,feature_i])
    Xy1=Xy[Xy[:,feature_i]!=door,:]
    Xy2=Xy[Xy[:,feature_i]==door,:]
    return Xy1,Xy2

class CARTtree:
    class node:
        def __init__(self, feature_i=None, door=None,
                     leaf_value=None, left=None, right=None):
            self.feature_i = feature_i#特征索引值
            self.door = door#特征划分阈值
            self.leaf_value = leaf_value#叶子节点取值(如果是叶子的话)
            self.left = left
            self.right= right
    def __init__(self, min_samples_split=2,mini_gini_impurity=999):
        self.root = None
        self.min_samples_split = min_samples_split
        self.mini_gini_impurity=mini_gini_impurity
    def build(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        best_criteria = None#初始化最佳特征索引和阈值
        best_sets = None#初始化数据子集
        Xy = np.concatenate((X, y), axis=1) #合并输入和标签
        n_samples, n_features = X.shape
        init_gini_impurity = 999
        if n_samples >= self.min_samples_split :
            for feature_i in range(n_features):
                unique_values = np.unique(X[:, feature_i])
                for door in unique_values:
                    # 特征节点二叉分裂
                    Xy1, Xy2 = feature_split(Xy, feature_i, door)
                    # 如果分裂后的子集大小都不为0
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # 获取两个子集的标签值
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self.gini_impurity_calculate(y, y1, y2)

                        #获取最小基尼不纯度and最佳特征索引和分裂阈值
                        if impurity < init_gini_impurity:
                            init_gini_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "door": door}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": Xy1[:, n_features:],
                                "rightX": Xy2[:, :n_features],
                                "righty": Xy2[:, n_features:]
                            }
        if init_gini_impurity < self.mini_gini_impurity:
            left_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"])
            right_branch = self._build_tree(best_sets["rightX"], best_sets["righty"])
            curNode=self.node(feature_i=best_criteria["feature_i"], door=best_criteria["door"], left=left_branch, right=right_branch)
            return curNode
        #计算叶子计算取值
        leaf_value = self.leaf_value_calculate(y)
        return self.node(leaf_value=leaf_value)

    def predict_value(self, x,tree=None):#对一个向量
        if tree==None:
            tree = self.root
        if tree.leaf_value is not None:
            return tree.leaf_value
        feature_value = x[tree.feature_i]
        #判断落入左子树还是右子树
        branch = tree.left
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.door:
                branch = tree.left
        elif feature_value == tree.door:
            branch = tree.right
        return self.predict_value(x,branch)#递归

    def predict(self, X):#对矩阵
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred
    def gini_impurity_calculate(self, y, y1, y2):
        p = len(y1) / len(y)
        gini_impurity = p *gini(y1) + (1-p) *gini(y2)
        return gini_impurity
    def leaf_value_calculate(self,y):
        unique=np.unique(y)
        y_=[i for i in y]
        maxlen=0
        answer=None
        for i in unique:
            if y_.count(i)>maxlen:
                maxlen=y_.count(i)
                answer=i
        return answer


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import datasets
data = datasets.load_iris()
X, y = data.data, data.target
y=y.reshape((-1,1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = CARTtree()
clf.build(X_train, y_train)
y_pred = clf.predict(X_test)

accu=np.sum(np.array(y_pred)==np.array(y_test))/np.array(y_pred).shape[1]
print(accu)
