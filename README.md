# Decision Tree Project
_Python implementation of decision tree for classification of meteorological dataset, includes pruning and ensemble methods_

## 1. Models and Methods
### 1.1 Decision Tree Models
Here I used a improved CART model as my basic decision tree model, which would split dataset and choose best feature based on __Gini__ or __Entropy__.

### 1.2 Pruning Methods
Several tree pruning methods were implemented in class `DecisionTree` to avoid overfitting.
1. __Reduced Error Pruning__
2. __Pessimistic Pruning__
	Unlike other pruning methods, pessimistic pruning is a top-down algorithm, which is normally done by going through the nodes from the top of the tree. Here I also used a bottom-up method, which was brought up in a lecture from USC years ago (Machine Learning CSCI-567).
3. __Minimum Error Pruning__

### 1.3 Ensemble Methods
For now, I just use the ensemble methods providied in [Scikit-Learn](http://scikit-learn.org/stable/modules/ensemble.html#bagging).
1. __AdaBoost__
2. __Bagging__
3. __Random Forest__

### 1.4 QnA
__Q__: Why is it recommended not to prune the trees while training random forest / bagging?  
__A__: Pruning methods are usually used to prevent overfitting. As random forests do sampling with replacement along with random selection of features at each node for splitting the dataset, the correlation between the weak learners (individual tree models) would be low. So generally random forests can do a great job with just full depth. As for bagging, only __variance__ can be reduced through the bagging process, not bias (we can see high bias as underfitting and high variance as overfitting, see [bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)). So we'd like the individual trees to have __lower bias__, in which case, overfitting trees are more than suitable.

## 2. Dataset
The dataset contains hourly meteorological data from the paper _Assessing Beijing's PM 2.5 pollution: severity, weather impact, APEC and winter heating_.
