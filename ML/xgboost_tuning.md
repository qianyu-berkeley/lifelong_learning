# XGboost Fine Tuning Steps

The notes to explain the steps to Fine tuning of XGBoost Models on Tabular Data


## XGBoost Advantage

* XGboost has regularization to reduce overfitting
* It can enable parallel processing (see [Ref1](https://zhanpengfang.github.io/418home.html) and [spark scalar based xgboost](https://docs.databricks.com/machine-learning/train-model/xgboost-spark.html))
* handeling missing value
* Tree Pruning: Unlike GBM, XGBoost makes splits up to the max_depth specified and then starts pruning the tree backward and removing splits beyond which there is no positive gain
* Build-in Cross-Validation
* Can continue on an existing model from previous iterations

## Parameters

### General parameters:

* **booster [default=gbtree]**
  * Select the type of model to run at each iteration. It has 2 options:
  * gbtree: tree-based models
  * gblinear: linear models
* **silent[default=0]**:
  * if set to 1, no running message will be printed
* **nthread[default to max num of threads (core) available]**
  * the algorithm will detect automatically, one can change if we do not want to use all the cores

### Booster Parameters:

* **eta[default=0.3]|learning_rate if use ScikitLearn api**
  * Analogous to the learning rate in GBM
  * Makes the model more robust by shrinking the weights on each step
  * Typical final values to be used: 0.01-0.2
* **min_child_weight[default=1]**
  * Defines the minimum sum of weights of all observations required in a child.
  * This is similar to min_child_leaf in GBM but not exactly. This refers to the min “sum of weights” of observations, while GBM has the min “number of observations”.
  * Used to control over-fitting. Higher values prevent a model from learning relations that might be highly specific to the particular sample selected for a tree.
  * Too high values can lead to under-fitting; hence, it should be tuned using CV.
* **max_depth [default=6]**
  * The maximum depth of a tree is the same as GBM.
  * Used to control over-fitting as higher depth will allow the model to learn relations very specific to a particular sample.
  * It should be tuned using CV.
  * Typical values: 3-10
* **max_leaf_nodes**
  * The maximum number of terminal nodes or leaves in a tree.
  * It can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
  * If this is defined, GBM will ignore max_depth.
* **gamma [default=0]**
  * A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
  * Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
* **max_delta_step [default=0]**
  * In the maximum delta step, we allow each tree’s weight estimation to be. If the value is set to 0, there is no constraint. If it is set to a positive value, it can help make the update step more conservative.
  * Usually, this parameter is not needed, but it might help in logistic regression when the class is extremely imbalanced.
This is generally not used, but you can explore further if you wish.
* **subsample [default=1]**
  * Same as the subsample of GBM. Denotes the fraction of observations to be random samples for each tree.
  * Lower values make the algorithm more conservative and prevent overfitting, but too small values might lead to under-fitting.
  * Typical values: 0.5-1
* **colsample_bytree [default=1]**
  * Similar to max_features in GBM. Denotes the fraction of columns to be random samples for each tree.
  * Typical values: 0.5-1
* **colsample_bylevel [default=1]**
  * Denotes the subsample ratio of columns for each split in each level.
  * I don’t use this often because subsample and colsample_bytree will do your job. but you can explore further if you feel so.
* **lambda [default=1]|reg_lambda if use Scikitlearn api**
  * L2 regularization term on weights (analogous to Ridge regression)
  * This is used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.
* **alpha [default=0]|reg_alpha if use Scikitlearn api**
  * L1 regularization term on weight (analogous to Lasso regression)
  * It can be used in case of very high dimensionality so that the algorithm runs faster when implemented
* **scale_pos_weight [default=1]**
  * A value greater than 0 should be used in case of high-class imbalance as it helps in faster convergence.

### Learning Task Parameters

* **objective [def:w
* ault=reg:linear]**
  * This defines the loss function to be minimized. Mostly used values are:
  * binary: logistic –logistic regression for binary classification returns predicted probability (not class)
  * multi: softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
    * you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
  * multi: softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
* **eval_metric [ default according to objective ]**
  * The evaluation metrics are to be used for validation data.
  * The default values are rmse for regression and error for classification.
  * Typical values are:
    rmse – root mean square error
    mae – mean absolute error
    logloss – negative log-likelihood
    error – Binary classification error rate (0.5 thresholds)
    merror – Multiclass classification error rate
    mlogloss – Multiclass logloss
    auc: Area under the curve
* **seed [default=0]**
  * The random number seed.
  * It can be used for generating reproducible results and also for parameter tuning.


## General Approach for Parameter Tuning:

**Starting from big nobs and zoom into small nobs**

1. Choose a relatively high learning rate (e.g. 0.1), determine the optimum num of trees for the learning rate by leveraging xgboost cv function
2. Tune tree specific parameters for decided learning rate and num of trees
3. Tune regularization parameters
4. lower learning rate and decide optimial parameters

Step 1: Fix learning rate and estimator for tuning tree-based parameters
Step 2: Tune max_depth and min_child weight
Step 3: Tune Gamma
Step 4: Tune Subsample and colsample_bytree
Step 5: Tune regularization parameters
Step 6: Reduce learning rate








## Reference

* https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/