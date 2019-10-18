
# coding: utf-8

# In[287]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import xlrd
pd.set_option('display.max_columns',30)


# In[288]:


# Load datasets
train = pd.read_csv("PA1_train.csv")
dev = pd.read_csv("PA1_dev.csv")
test = pd.read_csv("PA1_test.csv")


# In[289]:


# Global parameters
eps = 0.5
np.random.seed(2)


# In[290]:


## Preprocessing

# Remove ID feature
train.drop('id',axis=1,inplace=True)
dev.drop('id',axis=1,inplace=True)
# Split the date
for df in [train,dev,test]:
    df['month'] = df.date.map(lambda x : x.split('/')[0])
    df['day'] = df.date.map(lambda x : x.split('/')[1])
    df['year'] = df.date.map(lambda x : x.split('/')[2]).map(int)
# Build tables (statistics)
num = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',       'view','sqft_above','sqft_basement','yr_built','yr_renovated'       ,'lat','long','sqft_living15','sqft_lot15']
cat = ['waterfront','condition','grade']
table1 = pd.DataFrame({'Feature':num,'Mean':train[num].mean(),              'Standard deviation':train[num].std(),             'Range':train[num].max()-train[num].min()})
d = {}
for i in cat:
    n = train[cat].nunique().max()-train[cat].nunique()[i]
    d[i.capitalize()+ " categories"] = list(train[i].value_counts(normalize=True).index)+[' ']*n
    d[i[0]+" %"] = list(train[i].value_counts(normalize=True)*100)+[' ']*n
table2 = pd.DataFrame(d)
#table1.to_excel('table1.xlsx',index=False)# in report
#table2.to_excel('table2.xlsx',index=False)# in report


# In[291]:


for df in [train,dev,test]:
    df.drop('date',axis=1,inplace=True)


# In[292]:


## Normalization
# We save copies of the non normalized datasets for part 3
train_raw = train.copy()
dev_raw = dev.copy()
test_raw = test.copy()


# In[338]:


# making sure we have numerical features for part 3 :
for df in [train_raw,dev_raw]:
    for col in df.columns:
        df[col] = df[col].map(float)


# In[293]:


cols = list(train.columns)
cols.remove('price')
cols.remove('dummy')
for col in cols:
    train[col] = train[col].map(float)
    dev[col] = dev[col].map(float)
    test[col] = test[col].map(float)
    M = train[col].max()
    m = train[col].min()
    train[col] = train[col].map(lambda x : (x-m)/(M-m))
    dev[col] = dev[col].map(lambda x : (x-m)/(M-m))
    test[col] = test[col].map(lambda x : (x-m)/(M-m))
train.price = train.price.map(float)
M = train.price.max()
m = train.price.min()
train['normalized_price'] = train.price.map(lambda x : (x-m)/(M-m))


# In[294]:


def learn(train,dev,gamma,max_it,lambdA):
    t = time.time()
    y = np.transpose(np.array(train.normalized_price,ndmin=2))
    y_raw = np.transpose(np.array(train.price,ndmin=2))
    X = np.array(train.drop(['price','normalized_price'],axis=1))
    features = list(train.drop(['price','normalized_price'],axis=1).columns)
    N = X.shape[0]
    d = X.shape[1]
    w = np.random.rand(d,1)
    
    norm_grad = 100
    SSE = []
    c = 0
    while (norm_grad > eps) & (c < max_it):
        error = X.dot(w) - y
        grad = 2*np.transpose(X).dot(error)+2*lambdA*w
        norm_grad = np.linalg.norm(grad)
        w -= gamma*grad
        SSE.append(np.linalg.norm(((X.dot(w))*(M-m)+m) - y_raw)**2)
        c += 1
    # SSE validation :
    y_dev = np.transpose(np.array(dev.price,ndmin=2))
    X_dev = np.array(dev.drop('price',axis=1))
    SSE_dev = X_dev.dot(w)
    SSE_dev = np.linalg.norm(((X_dev.dot(w))*(M-m)+m) - y_dev)**2
    
    # Mean Relative Absolute Error on validation :
    MRAE = np.round((pd.Series((((X_dev.dot(w))*(M-m)+m) - y_dev)[:,0]).map(abs)/dev.price).mean(),4)
    
    elapsed = time.time() - t
    return (w,SSE,c,elapsed,SSE_dev,MRAE,features)


# In[295]:


## Part 1 :
lambdA = 0
max_it = 500000 # we tried with 1M5 for 1e-7 but norm_grad won't get lower
gammas = [1e-0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
all_c = []
final_SSEs_train = []
SSEs_validation = []
all_MRAE = []
SSE_train_curves = []
all_w = []
for gamma in gammas:
    temp = learn(train,dev,gamma,max_it,lambdA)
    SSE_train_curves.append(temp[1])
    all_c.append(temp[2])
    final_SSEs_train.append(temp[1][-1])
    SSEs_validation.append(temp[4])
    all_MRAE.append(temp[5])
    all_w.append(temp[0])
    features = temp[-1]


# In[297]:


results_part1 = pd.DataFrame({'Gamma':gammas,'iterations':all_c,'SSE training':final_SSEs_train,             'SSE validation':SSEs_validation,'MRAE':all_MRAE})
#results_part1.to_excel('results_part1.xlsx',index=False)# in report


# In[298]:


# Saving curves in .csv files to plot them later
#temp = 'part1_gamma_1e-'
#c = 0
#for i in SSE_train_curves:
#    name = temp+str(c)
#    pd.DataFrame({name:i}).to_csv(name+'.csv',index=False)
#    c += 1


# In[299]:


# Learning weights from the best solution :
# the one from gamma = 1e-5
weights = pd.DataFrame({'Features':features,'Weights':list(all_w[5][:,0])})
weights = weights.sort_values(by='Weights',ascending=False)


# In[300]:


#weights.to_excel('part1_best_w.xlsx',index=False)# in report


# In[301]:


# figure in report :
#fig = weights.plot(x='Features',y='Weights',kind='bar',rot=80).get_figure()
#fig.set_size_inches(16,9)
#fig.savefig('part1_best_w.png', dpi=1000)


# In[311]:


## Part 2 :
max_it = 300000
gamma = 1e-5
lambdas = [0,1e-3,1e-2,1e-1,1,10,100]
SSE_train = []
SSE_validation = []
ws = []
for lambdA in lambdas:
    temp = learn(train,dev,gamma,max_it,lambdA)
    SSE_train.append(temp[1][-1])
    SSE_validation.append(temp[4])
    ws.append(temp[0])


# In[329]:


results_part2 = pd.DataFrame({'Lambda':lambdas,'SSE training':SSE_train,             'SSE validation':SSE_validation})
#results_part2.to_excel('results_part2.xlsx',index=False)# in report


# In[321]:


li = list(pd.Series(lambdas).map(lambda x : "lambda = "+str(x)))
d = {'Features':temp[-1]}
c = 0
for i in li:
    d[i] = list(ws[c][:,0])
    c += 1
part2_w_table = pd.DataFrame(d)


# In[330]:


#part2_w_table.to_excel('part2_w_table.xlsx',index=False)# in report


# ___

# ##### Hum...

# In[322]:


part2_w_table


# In[327]:


part2_w_table.iloc[:,1:].plot()


# ___

# In[ ]:


## Part 3 :


# In[341]:


def learn_part_3(train,dev,gamma,max_it):
    y = np.transpose(np.array(train.price,ndmin=2))
    X = np.array(train.drop('price',axis=1))
    N = X.shape[0]
    d = X.shape[1]
    w = np.random.rand(d,1)
    
    norm_grad = 100
    SSE = []
    SSE_dev = []
    c = 0
    while (norm_grad > eps) & (c < max_it):
        error = X.dot(w) - y
        grad = 2*np.transpose(X).dot(error)
        norm_grad = np.linalg.norm(grad)
        w -= gamma*grad
        SSE.append(np.linalg.norm(X.dot(w) - y)**2)
        c += 1
        # SSE validation :
        y_dev = np.transpose(np.array(dev.price,ndmin=2))
        X_dev = np.array(dev.drop('price',axis=1))
        SSE_dev.append(np.linalg.norm(X_dev.dot(w) - y_dev)**2)
        
    return (w,SSE,c,SSE_dev)


# In[358]:


max_it = 10000
gammas = [1,0,1e-3,1e-6,1e-9,1e-15]
SSEt = []
SSEv = []
for gamma in gammas:
    temp = learn_part_3(train_raw,dev_raw,gamma,max_it)
    SSEt.append(temp[1])
    SSEv.append(temp[3])


# In[368]:


li = list(pd.Series(gammas).map(lambda x : 'part3_gamma_'+str(x)))
c = 0
for i in li:
    table = pd.DataFrame({'SSE training':SSEt[c],'SSE validation':SSEv[c]})
    c += 1
    #table.to_csv(i,index=False)# curves in the report


# In[367]:


plt.plot(np.log10(table.iloc[:,0]))

