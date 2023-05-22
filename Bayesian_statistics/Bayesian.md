# Bayesian

## Bayesian Statistic

### Bayes Rule

* Probability distribution:
  * Marginal distributions: $p(A), p(B)$
    $$p(A) = \sum_{B}p(A, B)$$
    $$p(B) = \sum_{A}p(A, B)$$
  * Join distributions: $p(A, B)$
  * Conditional disbribution: $P(A|B), p(B|A)$
    $$p(A|B)=\frac{p(A,B)}{p(B)} == \frac{p(A,B)}{\sum_{A}p(A,B)}$$
    $$p(B|A)=\frac{p(A,B)}{p(A)} == \frac{p(A,B)}{\sum_{B}p(A,B)}$$
  * Derive Bayes Rule:
    $$p(A, B) = p(A|B)p(B) = p(B|A)p(A)$$
    $$p(B|A) = \frac{p(A, B)}{p(A)} = \frac{p(A|B)p(B)}{\sum_{B}P(A,B)}$$
    $$p(A|B) = \frac{p(A, B)}{p(B)} = \frac{p(A|B)p(B)}{\sum_{A}P(A,B)}$$
  * For continuous distribution, we would use integral instead of sum
* Gambler's Fallacy
  * If a randome event is independent, the past will not impact future probability. The future chance is always the same regardless of past winning of lossing
* The Monty Hall Problem

### Maxmum Likelihood Estimation

* Giving a set of data, fit the model to the data with model parameters that achieve the best fit.
* Bernoulli distribution (coil flip)
  * Discrete random variable
  * PMF (Probability mass function): x can be either 0 or 1
    $$p(x) = \theta^x(1 - \theta)^{1-x}$$
    $$p(x = 1) = \theta$$
    $$p(x = 0) = 1 - \theta$$
  * Maximum Likelyhood function (coin flip many times)
    * With $data = \{X1, X2, ... X_n\}$
    * liklihood is $L(\theta) = p(data | \theta) = \prod^N_{i}p(x_i|\theta) = \prod^n_{i}\theta^{x_i}(1-\theta)^{1-x_i}$
      * x is either 0 or 1, $\theta$ is the variable 
    * We are trying to solve $\theta$ so we can maximizing the $L$ likelihood
    * $\frac{dL}{d\theta} = 0\ or\ \hat{\theta} = argmaxL(\theta)$
    * Bernoulli likihood is not the same is binomial distribution which consider order of events


## Bayesian Machine Learning

* The Bayesian Approach
  * "Everything is a random variable"
  * e.g. Gaussian distribution with $\mu$ and $\sigma$
    * Frequentist approach treat $\mu$ and $\sigma$ as a number calculated from the samples
    * Bayesian appraoch treat $\mu$ and $\sigma$ as a random variable, we try to find $p(\mu, \sigma^2 | X)$ (find distribution instead of number)
* In the machine learning context
  * e.g. Lineary Regression $y = W^Tx$, instead of finding a $W$, we try to solve for the distribution of $w$, if x, y is training data, we solve for $p(W | x, y)$
* Bayesign Network
  * Bayes nets are a general model
  * We can model specific dependencies based on your understanding of the system
    * e.g. LDA (Latent Dirichlet Allocation) is a bayes net model

## Ref

* [Code Referece](https://github.com/lazyprogrammer/machine_learning_examples.git)