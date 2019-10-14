
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib as plt
pd.set_option('display.max_columns',30) #to see all the columns


# In[2]:


train = pd.read_csv("PA1_train.csv")


# # Part 0

# ## (a)

# In[3]:


# Removing the ID feature :
train.drop('id',axis=1,inplace=True)


# #### Why do you think it is a bad idea to use this feature in learning?
# ___
# Our goal is to predict the price of a house given its features. The index of the house in the database is not related to its value, that feature came from the extraction process. Using it in learning would increase the complexity of the model for no reason and might cause overfitting.

# ## (b)
# #### Split the date feature into three separate numerical features: month, day , and year. Can you think of better ways of using this date feature?

# In[4]:


train['month'] = train.date.map(lambda x : x.split('/')[0])
train['day'] = train.date.map(lambda x : x.split('/')[1])
train['year'] = train.date.map(lambda x : x.split('/')[2])


# Suggestion of ways to use this date feature :
# - We could compute the number of days between today and the date the house was sold, that would give us a numerical feature that may partly explain the price of the house.
# - We could also just keep the `year`, which would be a simpler and categorical version of the first suggestion.
# - We could create a new feature : the number of years between the building year and the year it was sold.

# ## (c)
# #### Build a table that reports the statistics for each feature. For numerical features, please report the mean, the standard deviation, and the range. Several of the features (waterfront, grade, condition (the later two are ordinal)) that are marked numeric are in fact categorical. For such features, please report the percentage of examples for each category.

# In[5]:


num = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',       'view','sqft_above','sqft_basement','yr_built','yr_renovated'       ,'lat','long','sqft_living15','sqft_lot15']
cat = ['waterfront','condition','grade']


# In[6]:


pd.DataFrame({'Feature':num,'Mean':train[num].mean(),              'Standard deviation':train[num].std(),             'Range':train[num].max()-train[num].min()})


# In[14]:


import matplotlib as plt


# In[17]:


train.sqft_living.hist()


# In[15]:


train.sqft_living.plot.density()


# In[11]:


train.sqft_living.describe()


# In[7]:


d = {}
for i in cat:
    n = train[cat].nunique().max()-train[cat].nunique()[i]
    d[i.capitalize()+ " categories"] = list(train[i].value_counts(normalize=True).index)+[' ']*n
    d[i[0]+" %"] = list(train[i].value_counts(normalize=True)*100)+[' ']*n
pd.DataFrame(d)


# In[9]:





# ## (d)
# #### Based on the meaning of the features as well as the statistics, which set of features do you expect to be useful for this task? Why?

# Some of them ...

# ## (e)
# #### Normalize all features to the range between 0 and 1 using the training data. Note that when you apply the learned model from the normalized data to test data, you should make sure that you are using the same normalizing procedure as used in training.

# $$Normalize(x)=\frac{x-min(x)}{max(x)-min(x)}$$

# In[24]:


for i in train.drop(['dummy','date'],axis=1).columns: #not possible to normalize date so we drop it, also dummy
    train[i] = train[i].map(float)
    M = train[i].max()
    m = train[i].min()
    train[i] = train[i].map(lambda x : (x-m)/(M-m))


# In[ ]:


"""

Multiple lines comment

muefzefzef
zefzefzef
zefzef

"""


# # Part 1

# $$
# \nabla L(w) = \sum_{i=1}^N (w^T x_i - y_i)x_i
# $$

# In[208]:


# design matrix : X
X = train.drop(['date'],axis=1)
X = np.mat(X)
N = X.shape[0] # number of rows, 10000
np.random.seed(42)
w = np.random.rand(X.shape[1],1)
y = X[:,-4]
eps = 0.5


# In[209]:


def grad(w,x,y,n):
    """computes the gradient of the loss function, all imputs are vectors"""
    s = np.repeat(0,23)
    for i in range(n):
        s = s+(np.matmul(np.transpose(w),np.transpose(X[i]))-y[i])*X[i]
    return np.transpose(s)


# In[211]:


grad(w,X,y,N)


# In[213]:


w = np.random.rand(X.shape[1],1)
norm_grad = 1
gamma = 1/1000
c = 0
while (norm_grad > eps)&(c<20):
    #print("1")
    gr = grad(w,X,y,N)
    w = w - gamma*gr
    norm_grad = np.linalg.norm(gr)**2/2
    print(norm_grad)
    c+=1


# In[178]:


norm_grad


# In[148]:


w


# In[135]:


np.linalg.norm(np.array([1,1]))


# In[107]:


X[0].shape


# In[112]:





# In[102]:


X[0].shape


# In[25]:


train


# In[53]:


np.transpose(np.mat([[0,1],[2,3]]))


# In[56]:


X = np.mat([[0,1],[2,3]])
X[0]


# In[57]:


for i in range(5):
    print(i)


# In[85]:


w=np.array([1,2])
np.matmul(np.transpose(w),w)


# In[90]:


def grad(w,x,y,n):
    """computes the gradient of the loss function, all imputs are vectors"""
    s = np.zeros(n)
    for j in range(n):
        s = s+np.matmul((np.matmul(x[j],np.transpose(w))-y[j]),x[j])
    return s


# In[91]:


# design matrix : X
X = train.drop(['date'],axis=1)
X = np.mat(X)
N = X.shape[0] # number of rows, 10000
np.random.seed(42)
w0 = np.random.rand(N,1)
y = X[:,-4]
X[0].shape
grad(w0,X,y,N)


# In[78]:


.shape
#grad(w0,X,y,N)


# In[30]:


X[0,1]


# In[22]:


w0 = np.array([0,0])
w0


# In[ ]:


SSE = []
eps = 0.5


# In[18]:


train

