# House-Price-Prediction

## Content

 - [Objective](#obj)
 - [Data](#data)
 - [Part 0 : Exploratory Data Analysis](#Part-0)
 - [Part 1 : Impact of different Learning Rate](#Part-1)
 - [Part 2 : Impact of Regularization hyperparameter](#Part-2)
 - [Part 3 : Impact of Normalization](#Part-3)
 - [Results](#Results)
 - [Conclusion](#Conclusion)
---

<details>
<summary> <a name="obj"><b style="font-size:20px"> 
Objective</b> </a> </summary>

- To build a linear regression with L2 regularization that can be used to predict 
the house’s price based on a set of features.
- Develope insight on impact of Learning Rate and Regularization hyper-parameter 
  on model performance.
  
</details>




<details>
<summary> <a name="data"><b style="font-size:20px"> Data</b> </a> </summary>
The dataset consisted of historic data on houses sold between May 2014 to 
May 2015.There are two data files:  <b>training (10000 examples) and devlopment (5597 examples)</b> 

The dataset consisted of 23 features (including the dummy). 
The last one is the target for prediction. Variables Description Data Type
1. dummy [numeric]: 1
2. id [numeric]: a notation for a house 
3. date [string]: Date house was sold. __Splits into 3 categories: day of month,year, month__
4. bedrooms[numeric]: Number of Bedrooms/House
5. bathrooms[numeric]: Number of bathrooms/bedrooms 
6. sqft_living [numeric]: square footage of the home
7. sqft_lot [numeric]: square footage of the lot
8. floors [numeric]: Total floors (levels) in house
9. waterfront [numeric, Categorical]: House which has a view to a waterfront 
10. view [numeric]: Has been viewed 
11. condition [numeric, Categorical]: Overall condition 1 indicates worn out property and 5 excellent 
12. grade [numeric, Categorical]: Overall grade given to the housing unit. 1 poor ,13 excellent
13. sqft_above [numeric]: square footage of house apart from basement 
14. sqft_basement [numeric]: square footage of the basement 
15. yr_built [numeric] : Built Year
16. yr_renovated [numeric]: Year when house was renovated (0 if n/a)
17. zipcode [numeric]: zip 
18. lat Latitude [numeric] : coordinate 
19. long Longitude [numeric]: coordinate 
20. sqft_living15 [numeric]: Living room area in 2015 
21. sqft_lot15 [numeric]: lotSize area in 2015
22. price [numeric, continuous] : Price/100k, which is the __prediction target__
</details>

<details>
<summary> <a name="part-0"><b style="font-size:20px">
Part 0 : Exploratory Data Analysis</b> </a> </summary>

Preprocessing and simple analysis. Perform the following preprocessing of the your data.
</details>


<details>
<summary> <a name="part-1"><b style="font-size:20px"> 
Part 1 : Impact of different Learning Rate</b> </a> </summary>

Explore different learning rate for batch gradient descent. 
</details>


<details>
<summary> <a name="part-2"><b style="font-size:20px"> 
Part 2 : Impact of Regularization hyper-parameter</b> </a> </summary>

Experiments with dierent /lmbda values.
</details>


<details>
<summary> <a name="part-3"><b style="font-size:20px"> 
Part 3 : Impact of Normalization</b> </a> </summary>

 Training with non-normalized data Use the preprocessed data but skip the nor- malization.
</details>


<details>
<summary> <a name="results"><b style="font-size:20px"> 
Results</b> </a> </summary>

 ...
</details>

<details>
<summary> <a name="conclusion"><b style="font-size:20px"> 
Conclusion</b> </a> </summary>
 ...
</details>


