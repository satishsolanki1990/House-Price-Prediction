# EXECUTING THIS SCRIPT SHOULD BE ALRIGHT, 10 mins expected
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
pd.set_option('display.max_columns',30)

# Part 0 stuff necessary for part 1
train = pd.read_csv("PA1_train.csv")
train.drop('id',axis=1,inplace=True)
train['month'] = train.date.map(lambda x : x.split('/')[0])
train['day'] = train.date.map(lambda x : x.split('/')[1])
train['year'] = train.date.map(lambda x : x.split('/')[2])
for i in train.drop(['dummy','date'],axis=1).columns:
    train[i] = train[i].map(float)
    M = train[i].max()
    m = train[i].min()
    train[i] = train[i].map(lambda x : (x-m)/(M-m))

# Global parameters
eps = 0.5
np.random.seed(2)

def grad(w,X,y,d,N):
    s = np.zeros(d)
    for i in range(N):
        s += (np.transpose(w).dot(X[i])-y[i])*X[i]
    return np.transpose(np.array(s,ndmin=2))

def learn(drop_set,gamma,max_it):
    t = time.time()
    X = np.array(train.drop(drop_set,axis=1))
    N = X.shape[0]
    d = X.shape[1]
    y = np.transpose(np.array(train.price,ndmin=2))
    w = np.random.rand(d,1)
    
    SSE = [100]
    c = 0
    while (SSE[c] > eps) & (c < max_it):
        gr = grad(w,X,y,d,N)
        w -= gamma*gr
        SSE.append(np.linalg.norm(gr)**2)
        c += 1
    SSE = SSE[1:]
    
    elapsed = time.time() - t
    
    return(w,SSE,c,elapsed)


# Arguments from Satish latest script (see time result after execution)
'''
print("Runtime using Arguments from Satish latest script : ")
drop_set = ['price','sqft_living15','bathrooms','bedrooms','floors','lat','sqft_basement','waterfront','view','date','dummy',            'sqft_lot','condition','yr_built','yr_renovated','zipcode','long','sqft_lot15','month','day','year']
gamma = 10**(-4)
max_it = 5000
temp = learn(drop_set,gamma,max_it)
print("Learning took :\n"+str(temp[2])+" iterations\n"+      str(np.round(temp[3]))+" sec"+" ("+str(np.round(temp[3]/60,1))+" min)"     +"\nFinal SSE = "+str(temp[1][-1])+"\n")
'''

print("Let's take a bigger max_it and that's basically question (a) : ")
drop_set = ['date','price','waterfront','sqft_lot15','month','day']
max_it = 10
for gamma in [10**(0),10**(-1),10**(-2),10**(-3),10**(-4),10**(-5),10**(-6),10**(-7)]:
    temp = learn(drop_set,gamma,max_it)
    print("gamma = "+str(gamma))
    print("Learning took :\n"+str(temp[2])+" iterations\n"+      str(np.round(temp[3]))+" sec"+" ("+str(np.round(temp[3]/60,1))+" min)"     +"\nFinal SSE = "+str(temp[1][-1])+"\n")
