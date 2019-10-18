# EXECUTING THIS SCRIPT SHOULD BE ALRIGHT, 10 mins expected
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
pd.set_option('display.max_columns',30)

# Part 0 stuff necessary for part 1
# import training data
train = pd.read_csv("PA1_train.csv")
train.drop('id',axis=1,inplace=True)
train['month'] = train.date.map(lambda x : x.split('/')[0])
train['day'] = train.date.map(lambda x : x.split('/')[1])
train['year'] = train.date.map(lambda x : x.split('/')[2])
# Normalization of the training data (including price) except the date and dummy features
for i in train.drop(['dummy','date'],axis=1).columns:
    train[i] = train[i].map(float)
    M = train[i].max()
    m = train[i].min()
    train[i] = train[i].map(lambda x : (x-m)/(M-m))


# import test data and normalize it
dev = pd.read_csv("PA1_dev.csv")
dev.drop('id',axis=1,inplace=True)
dev['month'] = dev.date.map(lambda x : x.split('/')[0])
dev['day'] = dev.date.map(lambda x : x.split('/')[1])
dev['year'] = dev.date.map(lambda x : x.split('/')[2])
# Normalization of the test data (Excluding price) except the date and dummy features
for i in dev.drop(['dummy','date','price'],axis=1).columns:
    dev[i] = dev[i].map(float)
    M = dev[i].max()
    m = dev[i].min()
    dev[i] = dev[i].map(lambda x : (x-m)/(M-m))


# Global parameters
eps = 0.5
np.random.seed(2)
MaxP = train['price'].max()
minP = train['price'].min()
vMaxP = dev['price'].max()
vminP = dev['price'].min()

def SSEplots(SSE,gamma,lmbda):
    fig=plt.figure()
    x=[i for i in range(len(SSE)-1)]
    ax=fig.add_subplot(1,1,1)
    ax.plot(x,np.log(SSE[1:]),linewidth=3)
    plt.title('SSE vs no. of itereation for gamma '+str(gamma)+'lambda='+str(lmbda),fontweight='bold')
    plt.xlabel('No. of iteration', fontweight='bold')
    plt.ylabel('log(SSE)',fontweight='bold')
    plt.savefig('SSEplot_'+str(gamma)+'_lambda='+str(lmbda)+'.png')
    plt.show()
    return None
    

def dnorm(Nprice,m,M):
    DNprice=[]
    for i in range(len(Nprice)):
        DNprice.append(Nprice[i]*(M-m)+m)    
    return DNprice

def validation(w,dropset):
    X = np.array(dev.drop(drop_set,axis=1))
    # compute the new price
    y_actual= np.transpose(np.array(train.price,ndmin=2))
    y_new=np.matmul(X,w)
    y_newDN=dnorm(y_new,vminP,vMaxP)
    RSSE=np.linalg.norm(y_newDN-y_actual)
    return RSSE

def grad(w,X,y,d,N):
    s = np.zeros(d)
    for i in range(N):
        s += (np.transpose(w).dot(X[i])-y[i])*X[i]
    return np.transpose(np.array(s,ndmin=2))

def learn(drop_set,gamma,max_it,lmbda):
    t = time.time()
    X = np.array(train.drop(drop_set,axis=1))
    N = X.shape[0]
    d = X.shape[1]
    y = np.transpose(np.array(train.price,ndmin=2))
    w = np.random.rand(d,1)
    SSE=[100]
    c = 0
    while (SSE[c] > eps) & (c < max_it):
        gr = grad(w,X,y,d,N)
        w = w - gamma*gr - lmbda*w
        y_hatN= np.matmul(X,w)
        SSE.append(np.linalg.norm(gr)**2)
        c += 1
    y_hatDN=dnorm(y_hatN,minP,MaxP)
    DNRSSE=(np.matmul(np.transpose((y-y_hatDN)),(y-y_hatDN)))**0.5
    elapsed = time.time() - t  
    SSEplots(SSE,gamma,lmbda)
    return(w,SSE,c,elapsed,DNRSSE)

print("Let's take a bigger max_it and that's basically question (a) : ")
drop_set = ['date','price','waterfront','sqft_lot15','month','day']
max_it = 1000
# open a fine and write 
f=open('result.txt','w+')
for lmbda in [0,10**(-3),10**(-2),10**(-1),1,10,10**(2)]:
    for gamma in [10**(0),10**(-1),10**(-2),10**(-3),10**(-4),10**(-5),10**(-6),10**(-7)]:
        temp = learn(drop_set,gamma,max_it, lmbda)
        print("gamma = "+str(gamma)+'\t Lambda =' + str(lmbda))
        f.write("\ngamma = "+str(gamma)+'\t Lambda =' + str(lmbda))
        f.write("Weight Vector = "+str(temp[0]))
        print("Validation RSSE = "+str(validation(temp[0],drop_set)))
        f.write("Validation RSSE = "+str(validation(temp[0],drop_set)))
        print("iterations = "+str(temp[2]) +"\nTime = "+str(np.round(temp[3]))+"sec ("+ str(np.round(temp[3]/60,1))+"min)  " +"\nFinal SSE ="+str(temp[1][-1])+"\nFinal DNRSSE ="+str(temp[4])+"\n\n")
        f.write("iterations = "+str(temp[2]) +"\nTime = "+str(np.round(temp[3]))+"sec ("+ str(np.round(temp[3]/60,1))+"min)  " +"\nFinal SSE ="+str(temp[1][-1])+"\nFinal DNRSSE ="+str(temp[4])+"\n\n")
    
f.close()

