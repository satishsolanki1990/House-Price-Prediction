

# import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',30) #to see all the columns

# read the training data
train = pd.read_csv("PA1_train.csv")

"""
Part 0 
"""
# 0.(a)
train.drop('id',axis=1,inplace=True) # Removing the ID feature 

# 0.(b)
# split the date into month day and year 
train['month'] = train.date.map(lambda x : x.split('/')[0])
train['day'] = train.date.map(lambda x : x.split('/')[1])
train['year'] = train.date.map(lambda x : x.split('/')[2])

"""
Suggestion of ways to use this date feature :
 - We could compute the number of days between today and the date the house was sold, that would give us a numerical feature that may partly explain the price of the house.
 - We could also just keep the `year`, which would be a simpler and categorical version of the first suggestion.
 - We could create a new feature : the number of years between the building year and the year it was sold.
"""
#We could create a new feature : the number of years between the building year and the year it was sold.
# added a new feature yrs=(sold year - built year)
#train['yrs']=[int(i)-int(j) for i,j in zip(train['year'],train['yr_built'])



# 0.(c) Build a table that reports the statistics for each feature. For numerical features, 
num = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','view','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15']
cat = ['waterfront','condition','grade']

# Statistics for continuous features
df_num=pd.DataFrame({'Feature':num,'Mean':train[num].mean(), 'Standard deviation':train[num].std(),'Range':train[num].max()-train[num].min()})
#print("\n",df_num)

# Statistics for Categorical features
d = {}
for i in cat:
    n = train[cat].nunique().max()-train[cat].nunique()[i]
    d[i.capitalize()+ " categories"] = list(train[i].value_counts(normalize=True).index)+[' ']*n
    d[i[0]+" %"] = list(train[i].value_counts(normalize=True)*100)+[' ']*n
df_cat=pd.DataFrame.from_dict(d)
#print("\n",df_cat)


# 0.(d) Selected set of features 
# Answer: 
# generate heat map of Correlation values between features
plt.figure(figsize=(20,20))
drop_set =['sqft_living15','bathrooms','bedrooms','floors','lat','sqft_basement','waterfront','view','date','dummy','sqft_lot','condition','yr_built','yr_renovated','zipcode','long','sqft_lot15','month','day','year']
foo = sns.heatmap(train.drop(drop_set,axis=1).corr(), vmax=0.6, square=True, annot=True)


# 0.(e) $$Normalize(x)=\frac{x-min(x)}{max(x)-min(x)}$$

for i in train.drop(drop_set,axis=1).columns: #not possible to normalize date so we drop it, also dummy
    train[i] = train[i].map(float)
    M = train[i].max()
    m = train[i].min()
    train[i] = train[i].map(lambda x : (x-m)/(M-m))

"""
part 1 
fix lambda to 0 
learning rate: 10^-0; 10^-1; 10^-2; 10^-3; 10^-4; 10^-5; 10^-6; 10^-7.
"""


# 1.(a) 

# Include all Features for training
# design matrix : X

X = train.drop(drop_set,axis=1)
X = np.mat(X)
gamma=10**(-4) # learning Rate
eps=0.5 # convergence criteria
N = X.shape[0] # number of rows, 10000
np.random.seed(2) # random seed generation
w0 = np.random.rand(1,np.size(X[0])) # initialization of weight vector
S=[] # to store norm(s)
SSE=[] # to stroe SSE
y = X[:,-1] # output vector (House price)
counter=0
while (True):
    #computes the gradient of the loss function, all imputs are vectors
    s = np.zeros(np.size(X[0]))
    for j in range(N):
        s = s+(np.matmul(X[j],np.transpose(w0))-y[j])*X[j]
    S.append(np.linalg.norm(s))
    w0=w0-gamma*s
    
    # y_hat is pridiction 
    y_hat=np.matmul(X,np.transpose(w0))
    SSE.append(np.linalg.norm(y_hat-y)**2)
    counter=counter+1
    print(np.linalg.norm(s))
    if counter>5000:
        print("counter error")
        break
    if np.linalg.norm(s)<eps:
        break


# 1.(b) Report the SSE on the training data and the validation data, 
# number of iterations needed to achieve the convergence condition for training.


"""
(c) Use the validation data to pick the best converged solution, and report the learned weights for each
feature. Which feature are the most important in deciding the house prices according to the learned
weights? Compare them to your pre-analysis results (Part 0 (d)).
"""


"""
Part2 (30 pts). Experiments with di
erent lambda values. For this part, you will test the effect of the
regularization parameter on your linear regressor. Please exclude the bias term from regularization. It is
often the case that we don't really what the right  value should be and we will need to consider a range of
di
erent  values. For this project, consider at least the following values for lambda: 0; 10^-3; 10^-2; 10^-1; 1; 10; 100.
Feel free to explore other choices of  using a broader or 
finer search grid. Report the SSE on the training data
and the validation data respectively for each value of lambda. Report the weights you learned for di
erent values
of lambda. What do you observe? Your discussion of the results should clearly answer the following questions:
(a) What trend do you observe from the training SSE as we change  value?
(b) What tread do you observe from the validation SSE?
(c) Provide an explanation for the observed behaviors.
(d) What features get turned off
 forlambda = 10, 10^-2 and 0 ?

Part 3 (10 pts). Training with non-normalized data Use the preprocessed data but skip the nor-
malization. Consider at least the following values for learning rate: 1, 0; 10^-3; 10^-6; 10^-9; 10^-15. For each
value , train up to 10000 iterations ( Fix the number of iterations for this part). If training is clearly di-
verging, you can terminate early. Plot the training SSE and validation SSE respectively as a function of
the number of iterations. What do you observe? Specify the learning rate value (if any) that prevents the
gradient descent from exploding? Compare between using the normalized and the non-normalized versions
of the data. Which one is easier to train and why?
"""

