# Introduction
This is a practice competition at Kaggle(https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) in which the objective is to predict the sale price of houses using the given data about each house.

## Data
Data contains about 80 columns(ID column not included) with NaN and String values. Firstly, I changed all the String calues to categories using 
```
  data[column] = data[column].astype('category')
  data[column] = data[column].cat.codes
```
which turns different values in those columns to numbers starting from 1 and changes NaN to 0
Then, I used sklearn.preprocessing.StandardScaler to scale all values in the dataset to be smaller which improves the model accuracy.

## Methods
Because I wanted to see how each algorithm worked in this situation instead of choosing one of them, I tested as much of them as I could to see the differences and one important things that I learned from this experiment was that SGD only works well if the data is scaled while it is the exact opposite for Ransac which defiantly requires further research on my part to understand exactly why that is.

The best model from the normal algorithms was Huber which always gave the same accuracy at 0.84475 which also requires some research as other models’ accuracies changed every time the code ran (random_state = 42).

One other method that was used by one of the competitors was the catboost (CatBoost is a machine learning algorithm that uses gradient boosting on decision trees.) which does perform better when compared to Huber by 6% most of the time.

### Resources
1.	https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
2.	https://matplotlib.org/stable/tutorials/pyplot.html
3.	https://medium.com/analytics-vidhya/how-to-summarize-data-with-pandas-2c9edffafbaf
4.	https://seaborn.pydata.org/tutorial.html
5.	https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
6.	https://catboost.ai/en/docs/concepts/python-usages-examples (One submission used this method which performed better by 6% than Huber which was the best performing model that I tried.)
