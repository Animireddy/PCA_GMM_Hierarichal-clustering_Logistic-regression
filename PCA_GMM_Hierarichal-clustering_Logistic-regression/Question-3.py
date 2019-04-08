#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import math
import operator
import matplotlib
import matplotlib.pyplot as plt
from statistics import stdev,mean
import scipy.stats as ss
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,roc_curve,roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[14]:


k = 0
#k , l , m testing
def one_vs_all(data,y,epochs,lrate):
    classes = list(set(y))
    num_classes = len(classes)
    W = np.zeros((num_classes,data.shape[1]))
    k = 1
    print(data.shape[1])
    for itr in range(epochs):
        k = 2
        for c in range(num_classes):
            w = W[c]
            vecj = np.zeros(data.shape[1])
            k = 3
            for n in range(data.shape[0]):
                k = 4
                del_j = 0
                m = math.exp(np.matmul(data[n,:], np.transpose(w)))
                csum = 0
                for aux_c in range(num_classes):
                    k = 5
                    csum += math.exp(np.matmul(data[n,:] ,np.transpose(W[aux_c])))
                m /= csum
                del_j += m
                k = 6
                if(y[n] == classes[c]):
                    del_j -= 1


                l = 1    
                for i in range(data.shape[1]):
                    l = 2
                    vecj[i] += data[n,i]*del_j*lrate

            for i in range(len(w)):
                m = 1
                w[i] -= vecj[i]

                # normalizing w
            sqsum = 0
            for i in range(len(w)):
                sqsum += w[i]**2
            for i in range(len(w)):
                w[i] /= np.sqrt(sqsum)

            for i in range(len(w)):
                W[c][i] = w[i]

    return W, classes


# In[4]:


def Predict(inst,mean,std):
    inst = (inst-mean)/std
    inst1 = np.concatenate((np.array([1.0 for i in range(inst.shape[0])])[:, np.newaxis], inst), axis=1)
    inst2 = np.array(inst1,dtype=float)
    out = []
    for w in W:
        out.append(np.exp(np.dot(inst2,np.transpose(w))))
    return out
    


# In[5]:


wine = pd.read_csv("./wine-quality/data.csv")


# In[6]:


y_adm = wine.quality.values
X_adm = wine.iloc[:,:]
X_adm_train,X_adm_val,y_adm_train,y_adm_val=train_test_split(X_adm,y_adm,test_size=0.2,random_state=42)
X_train_temp = X_adm_train
X_adm_train = X_adm_train.drop('quality',axis = 1)
X_adm_val= X_adm_val.drop('quality',axis = 1)
X_adm_train=X_adm_train.values
X_adm_val=X_adm_val.values


# In[7]:



for C in list(set(y_adm)):
    print(C," ",wine[wine["quality"]== C].shape[0])


# In[8]:


Mean,Std=np.mean((X_adm_train),0),np.std((X_adm_train),0)
X_adm_train=(X_adm_train - Mean)/Std
X_adm_train = np.concatenate((np.array([1.0 for i in range(X_adm_train.shape[0])])[:, np.newaxis], X_adm_train), axis=1)
#X_adm_val = np.concatenate((np.array([1.0 for i in range(X_adm_val.shape[0])])[:, np.newaxis], X_adm_val), axis=1)


# In[9]:


W, classes = one_vs_all(X_adm_train,y_adm_train,100,0.5)


# In[11]:


out_Class = Predict(X_adm_val,Mean,Std)


# In[12]:


#print(np.array(out_Class))
Ind = np.argmax(out_Class, axis=0)
pred = []
for ind in Ind:
    pred.append(classes[ind])
#print(pred)


# In[13]:


print("F1-score",f1_score(y_adm_val,pred, average = 'micro'))
print("precision",precision_score(y_adm_val,pred, average = 'micro'))
print("Accuracy",accuracy_score(y_adm_val,pred))
print("Recall",recall_score(y_adm_val,pred, average = 'micro'))


# In[16]:


def logistic_regression(data,y,epochs,lrate):
    Temp_data = np.concatenate((np.array([1.0 for i in range(data.shape[0])])[:, np.newaxis], data), axis=1)
    theta = np.zeros(Temp_data.shape[1])
    theta[0] = 1
    x = 1
    for i in range(epochs):
        x = x+1
        z = np.dot(Temp_data,theta)
        h = 1 / (1 + np.exp(-z))
        gradient = np.dot(Temp_data.T, (h - y)) / y.size
        theta -= lrate * gradient
    return theta


# In[17]:


def predict(inst,theta,key,k,threshold,mean,std):
    #print(Mean.shape,inst.shape)
    inst = (inst-mean)/std
    inst1 = np.concatenate((np.array([1.0 for i in range(inst.shape[0])])[:, np.newaxis], inst), axis=1)
    inst2 = np.array(inst1,dtype=float)
    #print(inst.shape,theta.shape)
    prob = 1 / (1 + np.exp(-1*np.dot(inst2, theta)))
    Out = []
    v1 = 0
    for p in prob:
        v1 = v1 + 1
        if(p >= threshold and key>k):
            Out.append(key)
        elif(p < threshold and key<k):
            Out.append(key)
        else:
            Out.append(k)
    return Out


# In[18]:


def get_subtable(df, node,class1,class2):
    return df[operator.or_(df[node] == class1,df[node] == class2)].reset_index(drop=True)
    


# In[19]:


def one_vs_one(df,y,epochs,lrate):
    classes = list(set(y))
    num_classes = len(classes)
    Dict = {}
    for i in range(num_classes):
        Dict[classes[i]] = {}
        for j in range(i+1,num_classes): 
            temp = get_subtable(df,"quality",classes[i],classes[j]).values
            Dict[classes[i]][classes[j]] = logistic_regression(temp[:,:-1],temp[:,-1],epochs,lrate)
    return Dict
    


# In[20]:


def get_class(inst,threshold,mean,std):
    arr_class = []
    for key in Dict.keys():
        for k in Dict[key].keys():
            arr_class.append(predict(inst,Dict[key][k],key,k,threshold,mean,std))
    return np.array(arr_class)
    


# In[21]:


Dict = one_vs_one(X_train_temp,y_adm_train,100,0.5)


# In[22]:


arr_class = get_class(X_adm_val,0.4,Mean,Std)


# In[2]:


#print(arr_class)


# In[23]:


pred1 = []
for i in range(arr_class.shape[1]):
    (values,counts) = np.unique(arr_class[:,i],return_counts=True)
    ind=np.argmax(counts)
    #print(values[ind])
    pred1.append(values[ind])


# In[24]:


for i in range(X_adm_val.shape[0]):
    print(y_adm_val[i],pred1[i])


# In[25]:


print("F1-score",f1_score(y_adm_val,pred1, average = 'micro'))
print("precision",precision_score(y_adm_val,pred1, average = 'micro'))
print("Accuracy",accuracy_score(y_adm_val,pred1))
print("Recall",recall_score(y_adm_val,pred1, average = 'micro'))

