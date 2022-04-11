
import numpy as np
import pandas as pd
from math import log



def entropy(a):
    probs = []#a表中各个取值的概率分布
    for i in set(a):
        probs.append(a.count(i) / len(a))
    entropy=0
    for prob in probs:
        entropy+=-prob*log(prob, 2)
    return entropy

def spilt(df,col):#划分数据函数,返回字典，键为特征取值，值为一个矩阵
    #print(df[col].unique())
    unique_col=df[col].unique()
    dictionary={i:pd.DataFrame for i in unique_col}
    for key in dictionary.keys():
        dictionary[key]=df[:][df[col]==key]
    return dictionary

#a=spilt(df,'work')
#print(a)


def choose_best_character(df):
    entropy_D = entropy(df["answer"].tolist())
    characters=[]#除去答案的特征矩阵
    for col in df.columns:
        if col not in ["answer"]:
            characters.append(col)
            #print(col)
    max_infomation_gain, best_character = -9999, None
    max_splited = None
    for character in characters:
        splited_set = spilt(df, character)#划分后的字典
        entropy_DA = 0
        for character_value, subset in splited_set.items():
            entropy_Di = entropy(subset["answer"].tolist())
            entropy_DA += len(subset) / len(df) * entropy_Di
        infomation_gain = entropy_D - entropy_DA#信息增益

        if infomation_gain > max_infomation_gain:
            max_infomation_gain, best_character = infomation_gain, character
            max_splited = splited_set
    #print(best_character)
    #print(max_splited)
    return max_infomation_gain, best_character, max_splited


class ID3Tree:
    class Node:
        def __init__(self,name,leaf_value=None):
            self.name=name
            self.connetNodes={}
            self.leaf_value=leaf_value
        def connect(self,label,node):
            self.connetNodes[label]=node
    def __init__(self,df):
        self.characters=df.columns
        self.df=df
        self.root=self.Node("ROOT")

    def build(self,parent_node,parent_label,under_df,characters):
        max_information_gain,best_character, max_splited=choose_best_character(under_df[characters])
        #print(best_character)
        #print(max_splited)
        if not best_character:
            node=self.Node(under_df["answer"].iloc[0])
            parent_node.connect(parent_label,node)
        else:

            new_characters=[]
            for character in  characters:
                if character!=best_character:
                    new_characters.append(character)
            #预剪枝判断
            #先求剪枝后的正确率
            later_ratio=0
            ddf=[i for i in under_df["answer"]]
            unique_answer=under_df["answer"].unique()
            for i in unique_answer:
                if ddf.count(i)/len(ddf)>later_ratio:
                    later_ratio=ddf.count(i)/len(ddf)
                    correct_ans=i
            # 再求不剪枝的正确率
            total_num=0
            correct_num=0
            for spilted_data in max_splited.values():
                ddf = [i for i in spilted_data["answer"]]
                unique_answer = spilted_data["answer"].unique()
                maxcount=0
                for i in unique_answer:
                    if ddf.count(i)>maxcount:
                        maxcount=ddf.count(i)
                correct_num+=maxcount
                total_num+=len(ddf)
            current_ratio=correct_num/total_num
            #print(later_ratio)
            #print(current_ratio)
            if(later_ratio<current_ratio):#若剪枝导致正确率下降，则继续分支
                node = self.Node(best_character)
                parent_node.connect(parent_label, node)
                for spilted_value,spilted_data in max_splited.items():
                    self.build(node,spilted_value,spilted_data,new_characters)
            else:
                node = self.Node(best_character,correct_ans)
                parent_node.connect(parent_label, node)

    def buildTree(self):
        self.build(self.root,"root",self.df,self.characters)
    def predict_value(self, x,tree=None):#对一个向量
        if tree==None:
            tree = self.root
        if tree.leaf_value is not None:
            return tree.leaf_value
        feature_value = x[tree.name]

        branch = tree.connetNodes[feature_value]

        return self.predict_value(x,branch)#递归

    def predict(self, X):#对矩阵
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

df=pd.read_csv('./rent_data1.csv')
print((df['answer'].tolist()))
tree=ID3Tree(df)
tree.buildTree()
print(tree.root.connetNodes['root'].name)#"house"说明第一个节点判断是有无房子
print(tree.root.connetNodes['root'].connetNodes)
print(tree.root.connetNodes['root'].connetNodes['have'].leaf_value)
print(tree.root.connetNodes['root'].connetNodes['not'].name)
print(tree.root.connetNodes['root'].connetNodes['not'].connetNodes['not'].connetNodes)
#结果：
#house------(not)----->work----(not)---->False
#  |(have)               |(have)
#  -->True               -->True
