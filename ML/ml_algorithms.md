# Model Algorithms

## Regression

|    | Solving for $\mathbf{w}$| Gradient, $w^{n+1} = w^n - \eta \nabla_w$ | Loss Function |
|:---------|:----|:----|:----|
| OLS      | $\mathbf{w^T = (X^T X)^{-1} X^Ty}$ | $\frac{1}{n} \ \mathbf{X^T(y - Xw^T})$ | $\mathbf{\| y - Xw^T \|^2_2}$
| Linear Ridge    | $\mathbf{w^T = (X^T X + \lambda I)^{-1} X^Ty}$ | $\frac{1}{n} \  \mathbf{(X^TX + \lambda I) \ w^T - X^Ty}$ | $\mathbf{\| y - Xw^T \|^2_2 + \lambda \| w \|^2_2}$
| Linear Lasso    | Quadratic programming | | $\mathbf{\| y - Xw^T \|^2_2 + \lambda \| w \|_1}$
| Logistic | Newton-Raphson/GD | $\frac{1}{n} \ \mathbf{X^T(y - p)}$ | $- \sum_{i=1}^N y_i \log p_i + (1 - y_i) \log(1 - p_i)$ |

### Linear Regression

**Assumptions:**

1. $\epsilon_i$ are IID with from $\sim N(0, \sigma^2)$
    * **Homoscedasticity:** error terms must have constant variance
        * Breaking this assumption means that the Gauss-Markov theorem does not apply
        * Heteroscedasticity can cause standard errors of coefficients to be biased, though coefficent estimates won't be biased
    * **Exogeneity assumption:** $X_i$ were chosen by random by nature, indepedent from $\epsilon_i$ ("Endogeneity bias" = $\epsilon \not\!\perp\!\!\!\perp X$)
$$E(\hat \beta \mid X ) = \beta + \underbrace{(X^TX)^{-1}X^T E(\epsilon \mid X)}_{\text{if } X \perp \epsilon, \text{ then } E(\hat \beta \mid X) = \beta}$$
2. $Y_i$ was determined by the response schedule: $Y_{i,x} = x\beta + \epsilon_i$
3. No **multicollinearity** (i.e, no correlated regressors)
    * Multicollinearity can cause inflated coefficient standard errors, nonsensical coefficients, and a singular sample covariance matrix. 
    * Increased standard errors in turn means that coefficients for some independent variables may be found not to be significantly different from 0. In other words, by overinflating the standard errors, multicollinearity makes some variables statistically insignificant when they should be significant. 
    * Without multicollinearity, those coefficients might be significant. 
    * Use partial least squares or principal component regression.

**Least Squares derivation:**

$$
\begin{aligned}
RSS &=\| \mathbf{ y - X w } \|^2 = \sum^N_{i=1} (y_i - x_i^T w)^2\\
&= \mathbf{(y - Xw)^T(y-Xw)} \\
&= \mathbf{y^T y} - \underbrace{\mathbf{y^T X w - w^T X^T y}}_{\text{dim 1 x 1, = to its transpose}} + \mathbf{w^T X^T X w} \\
&= \mathbf{y^T y - 2w^T X^T y + w^T X^T X w} \\
\dfrac{\partial RSS}{\partial \mathbf{w}} &= \mathbf{-2X^Ty + 2X^TXw} = 0 \\
&= \underbrace{\mathbf{-2 X^T (y - X w)}}_{\text{derive from RSS w/ chain rule}} \\
\mathbf{w} &= (\mathbf{X^T X)^{-1} X^T y} \\
\dfrac{\partial^2 RSS}{\partial \mathbf{w} \partial \mathbf{w}^T} &= \mathbf{2X^TX}
\end{aligned}
$$

Note: $\mathbf{X}$ must have full column rank, and hence $\mathbf{X^TX}$ is positive definite.

**$R^2$, coefficient of determination:** measure of goodness of fit of linear model. The fraction (between 0 and 1) of MSE the model eliminates.

$$R^2 = \dfrac{\text{var}(X \hat \beta)}{\text{var}(Y)} = 1 - \dfrac{\sum_i (y_i - f_i)^2}{ \sum_i (y_i-\bar{y})^2 }$$

**Feature Transformations**

* **Log:** Performing a $\log_e$ transformation on $x$ (rather than $\log_{10}$) is directly interpertable as approximate proproptional differences: with a coefficient of 0.06, a differences of 1 in $x$ corresponds to an approximate 6% difference in $y$.
* **Square Root:** useful for compressing high values more mildly than by taking the log, but lack a clean interpretation; better for prediction tasks.

**Log-Log Models:** If we apply a log transformation to both features and outputs, we can interpret the coefficient as the expected proportional change in $y$ per proportional change in $x$.

### Regularization

**Lasso, $L_1$-norm:** similar to siBayesian regression with Laplace priors on the slopes

**Ridge, $L_2$-norm:** similar to Bayesian regression where slope priors are normally distributed with means equal to 0
### Linear Regression

* Assumptions:
  * A normal distribution of error terms
  * Independence in the predictors 
  * The mean residuals must equal zero with constant variance (Heteroscedasticity)
  * No correlation between the features


### [Logistic Regression](http://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf)

* Logistic Function

$$ h(x) = g(z) $$
$$ g(z) = \frac{1}{1+e^{-z}}$$
$$ z = \theta^Tx $$
$$
\begin{aligned}
\text{logit}(p) &= \log \bigg(\dfrac{1}{1-p}\bigg) = \beta_0 + \beta ^Tx \\ \text{logistic}(z) &= \text{logit}(z)^{-1} = \dfrac{1}{1 + e^{-z}} = \dfrac{e^z}{1 + e^z} \\
\end{aligned}
$$

$$
\begin{aligned}
\text{logit}(p) &= \log \bigg(\dfrac{1}{1-p}\bigg) = \beta_0 + \beta ^Tx \\ \text{logistic}(z) &= \text{logit}(z)^{-1} = \dfrac{1}{1 + e^{-z}} = \dfrac{e^z}{1 + e^z} \\
\end{aligned}
$$

It is a foundation algorithm for advanced machine learning/deep learning
* Assumptions
  * The dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal.
  * The observations to be independent of each other.
  * No correlation between the features
  * Assumes linearity of independent variables and log odds.  

#### Concept based on logistic regression

**Log Likelihood / Cross-Entropy**


$$\text{Let } y_i \in \{0, 1\}$$

$$p_i = \dfrac{1}{1+e^{- X_i \beta}}$$

$$
\begin{aligned}
\text{Likelihood}, L(\beta) &= \prod_{i=1}^n p_i^{y_i} (1-p_i)^{1-y_i} \\
\text{Log Likelihood}, l(\beta) &= \sum_{i=1}^N \big\{ y_i \log p_i + (1 - y_i) \log(1 - p_i) \big\} \\
&= \sum_{i=1}^N \log(1 - p_i) + \sum_{i=1}^N y_i \underbrace{( \log p_i - \log(1 - p_i) )}_{\log \left( \frac{x}{y} \right) = \log(x) - \log(y)} \\
&= \sum_{i=1}^N \log(1 - p_i) + \sum_{i=1}^N y_i \underbrace{\log\left( \dfrac{p_i}{1 - p_i}\right)}_{X_i \beta} \\
&= \sum_{i=1}^N - \log ( 1 + e^{X_i \beta} ) + \sum_{i=1}^N y_i(X_i \beta) \\
\end{aligned}
$$

$$\text{max(Log Likelihood) = min(Cross-Entropy / Negative Log Likelihood)}$$

**Derivative of Negative Log Likelihood**

$$
\begin{aligned}
\dfrac{\partial l(\beta)}{\partial \beta} &= - \left( - \sum_{i=1}^N \dfrac{1}{1+e^{-X_i \beta}} X_{i} + \sum_{i=1}^N y_i X_{i} \right) \\
&= - \sum_{i=1}^N (y_i - p_i) X_{i} \\
\text{In matrix form} &\rightarrow - \mathbf{X^T(y - p)} \\
&= - \mathbf{X^T(y - p)} + \underbrace{2 \lambda \beta}_{L_2} \\
\dfrac{\partial^2 l(\beta)}{\partial \beta \partial \beta^T} &= \mathbf{X^T W X}  \\
&= \mathbf{X^TWX} + \underbrace{2 \lambda \mathbf{I}}_{L_2} \\
\text{where } \mathbf{W} &= {\text{diag}(p_i (1 - p_i))} \\
\end{aligned}
$$


## Tree Based Model

### Meaure how good is the split?

* Intuitively: calculate std of the groups from split, find 2 group has the lowest std possible or weight average of std of 2 group => this is equal to minimize rmse
* officially: use object function to maximize information gain (`Gini`, `entropy`)

### General Property of Tree Based Models

* Can approximate complex nonlinear shapes
* Can capture complext interactions between
* Recursive partition of the input space
* Most trees are binary
* Missing data are treated as a level
* Unseen categorical level during inference are treated as N/A and classified with outliers
* Natural handle mix type of data 
* make no assumption of distribution of numerical features
* handle missing values
* handle multi-class outputs
* robust to outliers in input spaces
* computational scalable $O(nlogn)$
* cannot extract linear relationship

### Bagging vs Boosting

* Bagging: random subsample input space, build a set stronger learners for the sub-space, average results
* Boosting: build a set of weak learning, sequentially improve the model by prioritize weight to the mistake in the previous round, in the end all learners given a weight based on their accuracy, we consolidate a result

### Random Forrest (Bagging)

#### Why it is a perfect baseline algorithm for modern machine learning with large datasets?

* It make no assumption of the distribution of the features
* Works well with both numerical based categorical variable
* It handle outliers
* It is very robust towards overfitting even without validation set. We can use OOB (out of bag) sample to validate
* Handle a large number of features
* Require little feature engineering
* computational very efficient => embarrassingly parallel
* Most prevent high-variance with minor sacrifice of bias

#### How to handel N/A values in RF?

* Create a new binary feature call `feature_X_na`, set 1 for missing value, 0 otherwise
* Fill the missing value of the feature to -9999 if it has many N/A, use median if there is a small number of missing value

#### How does RF (Bagging) works?

* Random sample a subset of data (often with replacement or boostrapping) from training set
* Fit a deep tree, when splitting nodes during the construction of the tree, use random subset of features 
* The goal is to build a strong tree use subset of data/features to learn from a sub-space. Random sampling decrease variance
* We perform the above step many times in parallel to build many trees
* Take an average of those tree predictions as final prediction (Scikit-learn average probability prediction)
* **Intuition**: We build as accurate as possible trees in parallel for sub-sample and sub-features to learning, but there have very little correlation between them because of sampling. Therefore achieve low variance and better performance

#### What is ExtraTree?

* In extremely randomized trees (ExtraTreesClassifier and ExtraTreesRegressor classes), randomness goes one step further in the way splits are computed. 
* As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminate thresholds, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. 
* This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias

#### RF Parameters (in practice)

* `n_estimators`: 20 - 30 tree is a good number for feature learning (Add more tree will slow down the turn-around)
* `min_sample_leaf`: increase range in (1, 3, 5, 10, 25, 100) base on how large is the sample size
  - increase it will impact other parameters since tree depth inverse to sample size $\log(\text{sample size})$
    - reduce tree depth
    - reduce leaf notes
    - each estimator is less predictive but also less correlated
    - reduce overfitting, speed-up training
    - improve generalization
- `max_features`: determine the random sample of features (0.5 means half), (1, 0.5, log, sqrt)
  - increase cause each less accurate
  - tree varies more
  - reduce overfitting, improve generalization
  - sqrt is a good default

#### How is RF feature importance calculated?   

- Use the whole feature set to build random forest model and measure e.g. $RMSE$ of the tree prediction
- We pick one feature at a time
- randomly permutate the features column, apply to the same RF model check how much performance $RMSE$ get worse, get the difference
- Iterate all features to determine the importance of all features based on the performance drop of each feature

#### how to use feature importance insight?

- Talk to domain exports to understand top 10 - 15 features 
- The unit of importance is less important
- Interaction is also captured since we random sub-sample and sub-feature
- Take out low importance features and check whether the model performance drops
- Reduce redundancy may improve model => building a better tree, run faster
- If remove feature does not hurt performance or hurt slightly, we can remove them to make model simplier
  - Remove collinearity which splits importance to make feature importance plot cleaner and easy to understand
  - Help to determine which feature needs further feature engineering

#### Other tools

- Hierarchical clustering analysis on features
  - Exam spearmanr matrix and dendrogram
  - remove feature within group and check whether OOB_score get worse
  
- Partial dependancy analysis on features
  - plot partial depencency (**`pip install pdpbox`** python package)
  - Partial dependency analysis how important feature relates to target. It depicts the marginal effect of a feature on the target variable after accounting for the average effects of all other predictive features. It indicates how, holding all other variables except the feature of interest as they were, the value of this feature affects your prediction
  - How it partial dependency plot works? 
    -If all other feature is the hold constant, pick a feature of interests, replace the column with a fixed value, pass to RF, plot a prediction, and plot a point, replace with another value, plot a points and etc.
    - Simplified: take the average of all feature, and replace one feature with different value vs target, and plot 
  - **pdp* package also allow us to plot interaction, cluster features


### Adboost (Boosting)

- The whole idea of boost is to build small shallow trees (weak learner), analysis the residual, pay attention to mistake of previous round, give more weights to them in the next round
- Adboost following steps
  - 1. Draw a random subset of training data without replacement
  - 2. Feature is weighted for a stump (1 branch 2 leave tree)
  - 3. Assign higher weights (boost) to those misclassified samples and decrease correctly classified samples for the stumps in the previous step
  - 4. create new tree stumps
  - repeat steps 1, 2, 3, 4 until all sample fall into the right classes
- In Adboost, each stump's vote is weighted based on their prediction $accuracy = \frac{1}{2} \log \frac {1- \text{total error}}{\text {total error}}$

### Gradient Boost Machine (Boosting)

- Optimize the loss function of previous learner
- Optimize the residual of previous learner (predict residuals)
  - average + learning rate * prediction residual => take small step improve variance
- Additive model that regularize the loss function
- GBM (regression) follow the follow steps
  - Start with a single leave that make a prediction using average => avg
  - Build a tree on predict the residual (a little larger than stump, but still restrictive, max leave 8-32) where residual is called (pseudo residual)
  - build a tree to predict residuals and compbine with original leave. to avoid overfitting,  
  - If multiple samples fall into the same leaves, we just calculate the average. We add learning rate to scale the tree that learn from residual.
  - avg = avg + learning rate*residual prediction (learning rate between 0 and 1 for regularization to make variance low, otherwise we may overfit the tree 
  - We build another tree repeat the same steps on the updated avg
  - The next update avg = avg + 0.1 * first tree pred + 0.1 * second tree pred, then calculate residual ... chain of trees
  - we stop when we reach the define max tree number of base on early stop.

- GBM classification is very similar to GBM regression the key difference are
  - We calculate logistic function instead of average to figure out the initial predict probability
  - We calculate pseudo residual using 1 - prob or 0-prob i.e. observed class - predicted prob
  - output value from the residual decision need to be transformed instead of just use the value or average, we need to use a formula with probability.
    - sum of residual / (p1(1-p1) + p2(1-p2) ...)

### GBM Parameters (In practice)

- Types:
  - Tree parameters: affect individual tree in the model
  - Boosting parameters: affect boost operation
  - Other parameters
- Key Parameters:
  - `min_samples_split`: Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting.
Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. Too high values can lead to under-fitting hence, it should be tuned using CV.
  - `min_samples_leaf`: Defines the minimum samples (or observations) required in a terminal node or leaf. Used to control over-fitting similar to min_samples_split. Generally lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in majority will be very small.
  - `min_weight_fraction_leaf`: Similar to min_samples_leaf but defined as a fraction of the total number of observations instead of an integer. 
  - `max_depth`: The maximum depth of a tree. Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample. Should be tuned using CV.
  - `max_leaf_nodes`: The maximum number of terminal nodes or leaves in a tree. Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves. If this is defined, GBM will ignore max_depth.
  - `max_features`: The number of features to consider while searching for a best split. These will be randomly selected. As a thumb-rule, square root of the total number of features works great but we should check upto 30-40% of the total number of features. Higher values can lead to over-fitting but depends on case to case.
  - `learning_rate`: This determines the impact of each tree on the final outcome. GBM works by starting with an initial estimate which is updated using the output of each tree. The learning parameter controls the magnitude of this change in the estimates. Lower values are generally preferred as they make the model robust to the specific characteristics of tree and thus allowing it to generalize well. Lower values would require higher number of trees to model all the relations and will be computationally expensive.
  - `n_estimators`: The number of sequential trees to be modeled. GBM is fairly robust at higher number of trees but it can still overfit. This should be tuned using CV for a particular learning rate.
  - `subsample` The fraction of observations to be selected for each tree. Selection is done by random sampling. Values slightly less than 1 make the model robust by reducing the variance. Typical values ~0.8 generally work fine but can be fine-tuned further.
- Model tuning
  - Run use default parameter setting to create a baseline model and get performance metric
  - Set a high learning rate (0.05 ~ 0.2)
  - n_estimator (40 ~ 70) based on problem and computer resource
  - Tune tree parameters with Grids search
    - min_sample_split: 0.5-1% of total sample size
    - min_sample_leaf: 10 - 50 base on data
    - max_depth (5 - 10)
    - max_feature (sqrt)
    - subsample = 0.8
  - Lower the learning rate and increase estimator proportionally to achieve a robust model

### XGBoost

- Base on GBM with many improvement, e.g.
  - Add regularization
  - Parallel processing
  - Handling missing value
  - Tree pruning
- Parameters
  - `eta`: same as learning rate of GBM (0.01-0.2)
  - `min_child_weight` [default=1] Defines the minimum sum of weights of all observations required in a child. This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”. Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
  - `max_depth` [default=6] The maximum depth of a tree, same as GBM. Typical values: 3-10
  - `max_leaf_nodes`: same as GBM
  - `gamma` [default=0] A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split. Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
  - `max_delta_step` [default=0] In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
  - `subsample` [default=1] Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree. Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting. Typical values: 0.5-1
  - `colsample_bytree` [default=1] Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree. Typical values: 0.5-1
  - `colsample_bylevel` [default=1] Denotes the subsample ratio of columns for each split, in each level.
  - `lambda` [default=1] L2 regularization term on weights (analogous to Ridge regression) This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.
  - `alpha` [default=0] L1 regularization term on weight (analogous to Lasso regression) 
  - `scale_pos_weight` [default=1] A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.


### Unsupervised Learning

#### Commom usage

- Clustering
- Feature Extraction / dimension reduction
- Anomaly detection

#### Clustering

- K-means
  - 1. Determine number of clusters = k
  - 2. Randomly pick k centroid from sample points
  - 3. Assign each sample to the nearest centroid
  - 4. Taking the mean value of all of the samples (new center) and move the centroid to the new center of the samples that were assigned to it
  - 5. Repeat 3, 4 until the cluster assignment no long changes or a user defined tolerance.
  - Similarity measure: Euclidean distance
  - Model objective: minimize inertia aka. within-cluster sum of square errors.
  - If model canot not converge, we can change tolerance
  - K-means is the equivalent to the Expectation Maximization algorithm
  - To determine the best K: Use **elbow method** to find the optimal number of cluster (Hyper-parameter turning) by plotting number of cluster against distortion (`km.inertia_` or within-cluster SSE)

- Gaussian Mixture (Pobabilistic)

#### PCA 

- Dimension reduction of numerical features
- Improve computation efficiency with very small penalty in a model accuracy (curse of dimensionality)
- To identify patterns in data based on the correlation between features
- PCA aims to find the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions that the original one.
- Highlight sensitive to data scaling, need to standardize the features
- Eigenvectors of covariance matrix represent the principle components in the direction of maximum variance