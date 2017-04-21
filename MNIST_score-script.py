
# coding: utf-8

# In[1]:

import numpy as np
import scipy as sc

import sys

from matplotlib import pyplot as plt

from sklearn.preprocessing import normalize


nploader = np.load("allmnist.npz")

train = nploader['train']
train_labels= nploader['train_labels']
nploader.close()


# In[3]:

### Construct a W matrix 

### Lets start with a grid graph

### W should be 784 x 784 
a=np.zeros(784,dtype=np.int16)
a[1]=1
a[28]=1
a[-1]=1
a[-28]=1

W = sc.linalg.circulant(a)


# In[4]:

### Define the scoring function 

def distdiff(X,Y,W):
    
    totsum = 0
    for x in X:
        for y in Y:
            Wx = np.dot(W,x)
            Wy = np.dot(W,y)
            totsum += np.dot(Wx,Wy)
            
        
    
    return totsum
    
def distdiff_score(X,Y,W):
    S = distdiff(X,Y,W) / (distdiff(X,X,W) + distdiff(Y,Y,W))
    return S


# In[10]:

num_examples_per_class = int(sys.argv[1])

powermax = int(sys.argv[2])

labels_to_test = [float(sys.argv[3]),float(sys.argv[4])]

## Fetch two different classes 

train_X = (train[train_labels==labels_to_test[0]])
train_Y = (train[train_labels==labels_to_test[1]])

train_X_norm = normalize(train_X)
train_Y_norm = normalize(train_Y)


X = train_X_norm[:num_examples_per_class]
Y = train_Y_norm[:num_examples_per_class]

## loop over powers 
allpowers = np.arange(1,powermax+1)

scores_power = [distdiff_score(X,Y,np.linalg.matrix_power(W,curpow)) for curpow in allpowers]

plt.plot(allpowers,scores_power)
plt.xlabel('Power')
plt.ylabel('Score ')
plt.title('labels %d and %d' % (labels_to_test[0],labels_to_test[1]))
plt.xticks(allpowers)
plt.savefig('fig_%d_%d_powmax_%d_numex_%d.png' % (labels_to_test[0],labels_to_test[1],powermax,num_examples_per_class))
plt.close()

# In[ ]:



