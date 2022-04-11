
import numpy as np
from CART import CARTtree
from treeID3 import ID3Tree
def sample(X,y,n_of_classifiers):
    y_=y.reshape(-1,1)#y变为列向量
    number=X.shape[0]
    random_data=[]
    for i in range(n_of_classifiers):
        index=np.random.choice(number,number,replace=True)#有放回从number个数中随机取number个数
        random_X=X[index,:]
        random_y_=y_[index,:]
        random_data.append([random_X,random_y_])
    return random_data

def train(X,y,n_of_classifiers,feature_index):
    num_of_features=X.shape[1]
    random_data=sample(X,y,n_of_classifiers)
    for i in range(n_of_classifiers):
        X_,y_=random_data[i]
        index=np.random.choice(n_of_classifiers,15,replace=True)
        X_=X_[:,index]
        X_y_ = np.concatenate((X_, y_), axis=1)
        trees[i].build(X_, y_)
        feature_index.append(index)
        print('The {}th tree trained'.format(i + 1))

def predict_ans_for_list(trees,X,feature_index):
    pred_ans=[]
    for i in range(len(trees)):
        index=feature_index[i]
        X_=X[index]
        pred_ans.append(trees[i].predict_value(X_))
    max_num=0
    predict_y=None
    for i in set(pred_ans):
        if pred_ans.count(i)>max_num:
            max_num=pred_ans.count(i)
            predict_y=i
    return predict_y
def predict_ans_for_matrix(trees,X,feature_index):
    pred=[]
    for i in range(X.shape[0]):
        pred.append(predict_ans_for_list(trees,X[i,:],feature_index))
    return pred


feature_index=[]
n_estimators = 10
trees=[]
for i in range(n_estimators):
    tree=CARTtree()
    trees.append(tree)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
n_estimators = 10
max_features = 15
X, y = make_classification(n_samples=100, n_features=20, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
sampling_subsets = sample(X_train, y_train,n_estimators)

train(X_train, y_train,n_estimators,feature_index)
y_pred = predict_ans_for_matrix(trees,X_test,feature_index)
print(f"accuracy:{accuracy_score(y_test, y_pred)}")


