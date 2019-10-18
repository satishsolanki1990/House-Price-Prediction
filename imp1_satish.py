# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
pd.set_option('display.max_columns',30)

# Load datasets
train = pd.read_csv("PA1_train.csv")
dev = pd.read_csv("PA1_dev.csv")
test = pd.read_csv("PA1_test.csv")

# Global parameters
eps = 0.1
np.random.seed(2)

## Preprocessing
# Remove ID feature
train.drop('id',axis=1,inplace=True)
dev.drop('id',axis=1,inplace=True)
# Split the date
for df in [train,dev,test]:
    df['month'] = df.date.map(lambda x : x.split('/')[0])
    df['day'] = df.date.map(lambda x : x.split('/')[1])
    df['year'] = df.date.map(lambda x : x.split('/')[2])
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
table1.to_csv('table1.csv',index=False)# put in report
table2.to_csv('table2.csv',index=False)# put in report

# Feature engineering & feature selection a priori
drop_set = ['date','sqft_lot15','month','day','yr_renovated','waterfront']#0.2523
#drop_set = ['date']
for df in [train,dev,test]:
    # transforming yr_renovated into a binary variable because
    # so many houses have not been renovated
    df.yr_renovated = np.where(df.yr_renovated!=0,1,df.yr_renovated)
    df.drop(drop_set,axis=1,inplace=True)

## Normalization
# We save copies of the non normalized datasets for part 3
train_raw = train.copy()
dev_raw = dev.copy()
test_raw = test.copy()


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

def SSEplots(SSE,gamma,lmbda):
    fig=plt.figure()
    x=[i for i in range(len(SSE)-1)]
    ax=fig.add_subplot(1,1,1)
    ax.plot(x,(np.log(SSE[1:])/np.log(100)),linewidth=3)
    plt.title('SSE vs no. of itereation for gamma '+str(gamma)+' lambda='+str(lmbda),fontweight='bold')
    plt.xlabel('No. of iteration', fontweight='bold')
    plt.ylabel('log(SSE)',fontweight='bold')
    plt.savefig('SSEplot_'+str(gamma)+'_lambda='+str(lmbda)+'.png',dpi=1200)
    plt.show()
    return None
    

def learn(train,dev,gamma,max_it,lambdA):
    t = time.time()
    y = np.transpose(np.array(train.normalized_price,ndmin=2))
    y_raw = np.transpose(np.array(train.price,ndmin=2))
    X = np.array(train.drop(['price','normalized_price'],axis=1))
    d = X.shape[1]
    w = np.random.rand(d,1)
    
    norm_grad = 100
    SSE = []
    c = 0
    while (norm_grad > eps) & (c < max_it):
        error = X.dot(w) - y
        grad = 2*np.transpose(X).dot(error)+2*lambdA*w
        norm_grad = np.linalg.norm(grad)**2
        w -= gamma*grad
        SSE.append(np.linalg.norm(((X.dot(w))*(M-m)+m) - y_raw)**2)
        c += 1
    # SSE validation :
    y_dev = np.transpose(np.array(dev.price,ndmin=2))
    X_dev = np.array(dev.drop('price',axis=1))
    SSE_dev = X_dev.dot(w)
    SSE_dev = np.linalg.norm(((X_dev.dot(w))*(M-m)+m) - y_dev)**2
    
    print("Relative mean absolute error on validation : "+          str(np.round((pd.Series((((X_dev.dot(w))*(M-m)+m) - y_dev)[:,0]).map(abs)/dev.price).mean(),4)))
    print("iterations : "+str(c))
    print('SSE validation : '+str(SSE_dev))

    elapsed = time.time() - t
    SSEplots(SSE,gamma,lambdA)
    return (w,SSE,c,elapsed,SSE_dev)

max_it = 100000
# open a fine and write 
f=open('result.txt','w+')
for lambdA in [0,10**(-3),10**(-2),10**(-1),1,10,10**(2)]:
    for gamma in [10**(0),10**(-1),10**(-2),10**(-3),10**(-4),10**(-5),10**(-6),10**(-7)]:
        temp = learn(train,dev,gamma,max_it,lambdA)
        f.write("\ngamma = "+str(gamma)+'\t Lambda =' + str(lambdA))
        f.write("iterations : "+str(temp[2]))
        f.write("Time Elapsed = "+ str(temp[3])+" sec")
        f.write("Weight Vector = "+str(np.array(temp[0])))
        f.write("Validation SSE = "+str(temp[4]))
f.close()

