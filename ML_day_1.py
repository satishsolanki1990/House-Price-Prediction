import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import seaborn as sns


# Load datasets
train = pd.read_csv("PA1_train.csv")
dev = pd.read_csv("PA1_dev.csv")
test = pd.read_csv("PA1_test.csv")


## Preprocessing

# Remove ID feature
train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)
dev.drop('id',axis=1,inplace=True)
# Split the date
for df in [train,dev,test]:
    df['month'] = df.date.map(lambda x : int(x.split('/')[0]))
    df['day'] = df.date.map(lambda x : int(x.split('/')[1]))
    df['year'] = df.date.map(lambda x : int(x.split('/')[2]))

train.drop('date',axis=1,inplace=True)
# Build tables (statistics)
num = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement','yr_built','yr_renovated',
       'lat','long','sqft_living15','sqft_lot15']
cat = ['waterfront','condition','grade','view']
table1 = pd.DataFrame({'Feature':num,'Mean':train[num].mean(),'Standard deviation':train[num].std(),
                       'Range':train[num].max()-train[num].min()})
d = {}
for i in cat:
    n = train[cat].nunique().max()-train[cat].nunique()[i]
    d[i.capitalize()+ " categories"] = list(train[i].value_counts(normalize=True).index)+[' ']*n
    d[i[0]+" %"] = list(train[i].value_counts(normalize=True)*100)+[' ']*n
table2 = pd.DataFrame(d)
train['yr_diff']=train['year']-train['yr_built']
# print(train.columns)
# print(train.corr())
sns.heatmap(train.corr(method='spearman').round(1),annot=True)
plt.show()