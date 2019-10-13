

# import packages
import pandas as pd
import numpy as np

pd.set_option('display.max_columns',30) #to see all the columns

# read the training data
train = pd.read_csv("PA1_train.csv")


# Part 0


"""
(a) Remove the ID Feature. Why do you think it is a bad idea to use this feature in learning?
Answer:
    Our goal is to predict the price of a house given its features. The index of 
the house in the database is not related to its value, that feature came from 
the extraction process. Using it in learning would increase the complexity of 
the model for no reason and might cause overfitting.
"""

train.drop('id',axis=1,inplace=True) # Removing the ID feature 

"""
(b) split the date feature into three separate numerical features: month, 
day , and year. Can you think of better ways of using this date feature?
"""
train['month'] = train.date.map(lambda x : x.split('/')[0])
train['day'] = train.date.map(lambda x : x.split('/')[1])
train['year'] = train.date.map(lambda x : x.split('/')[2])

"""
Suggestion of ways to use this date feature :
 - We could compute the number of days between today and the date the house was sold, that would give us a numerical feature that may partly explain the price of the house.
 - We could also just keep the `year`, which would be a simpler and categorical version of the first suggestion.
 - We could create a new feature : the number of years between the building year and the year it was sold.
"""

"""
(c) Build a table that reports the statistics for each feature. For numerical features, 
please report the mean, the standard deviation, and the range. Several of the features 
(waterfront, grade, condition (the later two are ordinal)) that are marked numeric are 
in fact categorical. For such features, please report the percentage of examples for each category.
"""
num = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','view','sqft_above','sqft_basement','yr_built','yr_renovated'       ,'lat','long','sqft_living15','sqft_lot15']
cat = ['waterfront','condition','grade']

# Statistics for continuous features
df_num=pd.DataFrame({'Feature':num,'Mean':train[num].mean(), 'Standard deviation':train[num].std(),'Range':train[num].max()-train[num].min()})
print(df_num)

# Statistics for Categorical features
d = {}
for i in cat:
    n = train[cat].nunique().max()-train[cat].nunique()[i]
    d[i.capitalize()+ " categories"] = list(train[i].value_counts(normalize=True).index)+[' ']*n
    d[i[0]+" 
      %"] = list(train[i].value_counts(normalize=True)*100)+[' ']*n
df_cat=pd.DataFrame.from_dict(d)
print(df_cat)

"""
(d) Based on the meaning of the features as well as the statistics, which set of 
    features do you expect to be useful for this task? Why?
"""
# Answer: 

"""
(e) Normalize all features to the range between 0 and 1 using the training data. 
    Note that when you apply the learned model from the normalized data to test data,
    you should make sure that you are using the same normalizing procedure as used in training.

 $$Normalize(x)=\frac{x-min(x)}{max(x)-min(x)}$$
"""
for i in train.drop(['dummy','date'],axis=1).columns: #not possible to normalize date so we drop it, also dummy
    train[i] = train[i].map(float)
    M = train[i].max()
    m = train[i].min()
    train[i] = train[i].map(lambda x : (x-m)/(M-m))


# part 1 

"""
Explore different learning rate for batch gradient descent. For this part, you
will work with the preprocessed and normalized data and fix lambda to 0 and consider at least the following values
for the learning rate: 10^-0; 10^-1; 10^-2; 10^-3; 10^-4; 10^-5; 10^-6; 10^-7.
"""

"""
(a) Which learning rate or learning rates did you observe to be good for this particular dataset? What
learning rates make the gradient decent explode? Report your observations together with some 
example curves showing the training SSE as a function of training iterations and its convergence or
non-convergence behaviors.
"""


"""
(b) For each learning rate worked for you, Report the SSE on the training data and the validation data
respectively and the number of iterations needed to achieve the convergence condition for training.
What do you observe?
"""

"""
(c) Use the validation data to pick the best converged solution, and report the learned weights for each
feature. Which feature are the most important in deciding the house prices according to the learned
weights? Compare them to your pre-analysis results (Part 0 (d)).
"""

