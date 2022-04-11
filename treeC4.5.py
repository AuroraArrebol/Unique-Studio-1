

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
    max_infomation_gain_ratio, best_character = -9999, None
    max_splited = None
    for character in characters:
        splited_set = spilt(df, character)#划分后的字典
        entropy_DA = 0
        for character_value, subset in splited_set.items():
            entropy_Di = entropy(subset["answer"].tolist())
            entropy_DA += len(subset) / len(df) * entropy_Di
        infomation_gain = entropy_D - entropy_DA#信息增益
        #print(character)
        infomation_gain_ratio=infomation_gain/(entropy(df[character].tolist())+1e-4)

        if infomation_gain_ratio > max_infomation_gain_ratio:
            max_infomation_gain_ratio, best_character = infomation_gain_ratio, character
            max_splited = splited_set
    #print(best_character)
    #print(max_splited)
    return max_infomation_gain_ratio, best_character, max_splited


class ID3Tree:
    class Node:
        def __init__(self,name):
            self.name=name
            self.connetNodes={}
        def connect(self,label,node):
            self.connetNodes[label]=node
    def __init__(self,df):
        self.characters=df.columns
        self.df=df
        self.root=self.Node("ROOT")

    def build(self,parent_node,parent_label,under_df,characters):
        max_information_gain_ratio,best_character, max_splited=choose_best_character(under_df[characters])
        #print(best_character)
        #print(max_splited)
        if not best_character:
            node=self.Node(under_df["answer"].iloc[0])
            parent_node.connect(parent_label,node)
        else:
            node=self.Node(best_character)
            #print(node.name)
            parent_node.connect(parent_label,node)
            new_characters=[]
            for character in  characters:
                if character!=best_character:
                    new_characters.append(character)
            for spilted_value,spilted_data in max_splited.items():
                print(spilted_value)
                self.build(node,spilted_value,spilted_data,new_characters)
    def buildTree(self):
        self.build(self.root,"root",self.df,self.characters)



df=pd.read_csv('./rent_data1.csv')
#print((df['answer'].tolist()))
#print(entropy(df['credit'].tolist()))
tree=ID3Tree(df)
tree.buildTree()
print(tree.root.connetNodes['root'].name)#"house"说明第一个节点判断是有无房子
print(tree.root.connetNodes['root'].connetNodes)
dict=tree.root.connetNodes['root'].connetNodes
print(dict['have'])