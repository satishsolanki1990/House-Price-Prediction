# House-Price-Prediction

Content
=
 - [Objective](#objective)
 - [Data](#data)
 - [Part 0 : Exploratory Data Analysis](#Part-0--Exploratory-Data-Analysis)
 - [Part 1 : Impact of different Learning Rate](#Part-1--Impact-of-different-Learning-Rate)
 - [Part 2 : Impact of Regularization hyperparameter](#Part-2--Impact-of-Regularization-hyperparameter)
 - [Part 3 : Impact of Normalization](#Part-3--Impact-of-Normalization)
 - [Results](#Results)
 - [Conclusion](#Conclusion)
---

## Objective:

- To build a linear regression with L2 regularization that can be used to predict 
the house’s price based on a set of features.
- Develope insight on impact of Learning Rate and Regularization hyper-parameter 
  on model performance.

## Data:
The dataset consisted of historic data on houses sold between May 2014 to 
May 2015.There are two data files:  __training (10000 examples) and devlopment (5597 examples)__ 

The dataset consisted of 23 features (including the dummy). 
The last one is the target for prediction. Variables Description Data Type
- dummy: 1 numeric
- id: a notation for a house Numeric
- date: Date house was sold String
- bedrooms: Number of Bedrooms/House Numeric
- bathrooms: Number of bathrooms/bedrooms Numeric
- sqft_living: square footage of the home
- sqft_lot: square footage of the lot
- floors: Total floors (levels) in house
- waterfront: House which has a view to a waterfront Numeric view Has been viewed Numeric
- condition: Overall condition 1 indicates worn out property and 5 excellent grade Overall grade given to the housing unit. 1 poor ,13 excellent.
- sqft_above: square footage of house apart from basement sqft_basement square footage of the basementNumeric yr_built Built Year Numeric
- yr_renovated: Year when house was renovated (0 if n/a) zipcode zip Numeric
- lat Latitude: coordinate Numeric
- long Longitude: coordinate Numeric sqft_living15 Living room area in 2015 Numeric sqft_lot15 lotSize area in 2015 Numeric
- price: Price/100k, which is the prediction target


## Part 0 : Exploratory Data Analysis

Preprocessing and simple analysis. Perform the following preprocessing of the your data.

## Part 1 : Impact of different Learning Rate

   Explore different learning rate for batch gradient descent. 

## Part 2 : Impact of Regularization hyper-parameter

   Experiments with dierent /lmbda values.
 
## Part 3 : Impact of Normalization

 Training with non-normalized data Use the preprocessed data but skip the nor- malization.

## Results:



## Conclusion:


