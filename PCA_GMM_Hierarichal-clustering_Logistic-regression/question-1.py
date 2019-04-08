#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np 
import pandas as pd 
import os
import math
import operator
import matplotlib
import matplotlib.pyplot as plt
from statistics import stdev,mean
import scipy.stats as ss
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from scipy import linalg


# In[10]:


intrusion = pd.read_csv("./data.csv")


# In[11]:


def get_mean(data):
    return np.mean(data,0)


# In[12]:


def get_std(data):
    return np.std(data,0)


# In[13]:


def normalise(inst,mean,std):
    return (inst-mean)/(std)


# ### Function for PCA

# In[14]:


def PCA(X,k):
    a, b = np.shape(X)
    Mean = get_mean(X)
    Std = get_std(X)
    X = normalise(X,Mean,Std) # Nxd
    #print(X.shape,"   X")
    XT = np.transpose(X)
    #print(XT.shape,"   XT")   # dxN
    mat2 = np.cov(XT)
    #print(mat2.shape,"   mat2") # dxd
    eigenvalues, eigenvectors = np.linalg.eig(mat2)
    e = []
    val = 0
    vec = 0
    eigenvectors = np.transpose(eigenvectors)#kxe

        # sorting eigen vectors by values
    while(val < len(eigenvalues)):
        while(vec < len(eigenvectors)):
            e.append((eigenvalues[val], eigenvectors[vec]))
            val += 1
            vec += 1

    v=[]
    e.sort(key=lambda x: x[0], reverse=True)
    for tup in e:
        v.append(tup[1])
    #print(np.array(v).shape,"   v")
    a, b = np.shape(v)
    v = np.transpose(v)
    #print(np.array(v).shape,"   v")
    for i in range(a):
        s = 0
        for j in range(b):
            s += v[i][j]**2
        s = np.sqrt(s)
        for j in range(b):
            v[i][j] /= s
    p = v[:, :k]    # dxk
    #print(np.array(p).shape,"   p")
   
    z = np.matmul(X, p)  # Nxk
    #print(np.array(z).shape,"   z")
    return z,p


# ### Function for reconstruction of the data

# In[15]:


def reconstruction(red_data,V):
    V = np.transpose(V)
    #print(red_data.shape,V.shape)
    return np.dot(red_data,V)


# In[16]:


Data = intrusion.iloc[:,:-1].values
mean,std=np.mean(Data,0),np.std(Data,0)
Data=(Data - mean)/std


        
    


# ### Errors vs the selected number of dimensions

# In[17]:


T = 5
while T<=15:
    data_R = PCA(Data,T)
    data_N = reconstruction(data_R[0],data_R[1])
    error = 0
    div =  0
    for i in range(Data.shape[0]):
        error += np.sum((Data[i,:]- data_N[i,:])** 2)
        div += np.sum(Data[i,:] ** 2)
    print(T," ",error/div)
    T+=1


# In[18]:


df1 = PCA(Data,5)[0]


# In[19]:


clustering_method = []
purity = {}
for i in range(1,6):
    purity[i] = []


# ### Function for Kmeans

# In[20]:


def assignment(df, centroids,no_columns):
    #print(no_columns)
    k = 0
    for i in centroids.keys():
        k = k+1
        temp = np.zeros(df.shape[0])
        for j in range(no_columns):
            k = k+2
            #print(temp)
            temp = np.sum([temp,(df.iloc[:,j].values - centroids[i][j]) ** 2], axis = 0)
        temp = np.array(np.sqrt(temp))
        df['distance_from_{}'.format(i)] = pd.Series(temp)
    #print(temp)
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    #print(df['closest'])
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return df


# In[21]:


def update(df,centroids,no_columns):
    l = 0
    for i in centroids.keys():
        l=l+1
        for j in range(no_columns):
            l=l+2
            #print(df[df['closest']==i].iloc[:,j])
            centroids[i][j] = np.mean(df[df['closest'] == i].iloc[:,j])
    return centroids
    


# In[22]:


def k_means(df,no_clusters):
    no_columns = df.shape[1]
    np.random.seed(200)
    x = 0
    centroids = {
        i+1: df.iloc[np.random.randint(0,df.shape[0]),:].values
        for i in range(no_clusters)
    }
    
    df = assignment(df,centroids,no_columns)
    
    while True:
        x = x+1
        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(df,centroids,no_columns)
        df = assignment(df, centroids,no_columns)
        if closest_centroids.equals(df['closest']):
            break
    return df
        
    
    


# In[23]:



inp = {}
for index in range(5):
    inp[index] = df1[:,index]
Inp = pd.DataFrame(inp)
#print(Inp.shape)


# In[24]:



clusters = k_means(Inp,5)
clustering_method.append("KMeans")


# In[25]:


#print(clusters['closest'])


# In[26]:


clusters['xAttack'] = intrusion.iloc[:,-1].values


# In[27]:


# for i in range(intrusion.shape[0]):
#     print(clusters['closest'][i],clusters['xAttack'][i])


# In[28]:


Clus = {}
for i in range(1,6):
    Clus[i] = clusters[clusters['closest']==i]


# In[29]:


impurity = {}
for i in range(1,6):
    Maxi = 0
    for C in list(set(intrusion.iloc[:,-1].values)):
        DF = Clus[i]
        no = DF[DF['xAttack'] == C].shape[0]
        #print(no,Clus[i].shape[0])
        if(Maxi < no):
            Class = C
            Maxi = no
    impurity[i] = Maxi/Clus[i].shape[0]
    purity[i].append(impurity[i])
print(impurity)


# In[30]:


Inp['xAttack'] = intrusion.iloc[:,-1].values


# ### Agglomerative Clustering

# In[31]:


from sklearn.cluster import AgglomerativeClustering


# In[ ]:


clustering = AgglomerativeClustering(n_clusters =5).fit_predict(df1)
clustering_method.append("Agglomerative")


# In[ ]:


clustering


# In[ ]:


Inp['predict'] = np.array(clustering)+1


# In[ ]:


impurity = {}
for i in range(1,6):
    Maxi = 0
    DF = Inp[Inp['predict'] == i]
    for C in list(set(intrusion.iloc[:,-1].values)):
        no = DF[DF['xAttack'] == C].shape[0]
        #print(no,Clus[i].shape[0])
        if(Maxi < no):
            Class = C
            Maxi = no
    impurity[i] = Maxi/DF.shape[0]
    purity[i].append(impurity[i])
print(impurity)
    


# In[ ]:


from sklearn.mixture import GaussianMixture


# In[ ]:


GMM = GaussianMixture(n_components=5).fit_predict(df1)
clustering_method.append('GMM')


# In[ ]:


#print(GMM)


# In[ ]:


Inp['predict'] = np.array(GMM)+1


# In[ ]:


impurity = {}
for i in range(1,6):
    Maxi = 0
    DF = Inp[Inp['predict'] == i]
    for C in list(set(intrusion.iloc[:,-1].values)):
        no = DF[DF['xAttack'] == C].shape[0]
        #print(no,Clus[i].shape[0])
        if(Maxi < no):
            Class = C
            Maxi = no
    impurity[i] = Maxi/DF.shape[0]
    purity[i].append(impurity[i])
print(impurity)


# ### Bar diagram for purity measure vs clustering algorithms for 5 classes

# In[ ]:


index = np.arange(len(clustering_method))
plt.bar(index, purity[1])
plt.xlabel('clustering_method', fontsize=5)
plt.ylabel('purity', fontsize=8)
plt.xticks(index, clustering_method, fontsize=8)
plt.title('cluster 1')
plt.show()


# In[ ]:


index = np.arange(len(clustering_method))
plt.bar(index, purity[2])
plt.xlabel('clustering_method', fontsize=5)
plt.ylabel('purity', fontsize=8)
plt.xticks(index, clustering_method, fontsize=8)
plt.title('cluster 2')
plt.show()


# In[ ]:


index = np.arange(len(clustering_method))
plt.bar(index, purity[3])
plt.xlabel('clustering_method', fontsize=5)
plt.ylabel('purity', fontsize=8)
plt.xticks(index, clustering_method, fontsize=8)
plt.title('cluster 3')
plt.show()


# In[ ]:


index = np.arange(len(clustering_method))
plt.bar(index, purity[4])
plt.xlabel('clustering_method', fontsize=5)
plt.ylabel('purity', fontsize=8)
plt.xticks(index, clustering_method, fontsize=8)
plt.title('cluster 4')
plt.show()


# In[ ]:


index = np.arange(len(clustering_method))
plt.bar(index, purity[5])
plt.xlabel('clustering_method', fontsize=5)
plt.ylabel('purity', fontsize=8)
plt.xticks(index, clustering_method, fontsize=8)
plt.title('cluster 5')
plt.show()


# ### Part-5

# ## PCA can be applied on data with categorial variables as well by using label encoding which converts the categorical data into numerical data

# In[ ]:





# In[ ]:




