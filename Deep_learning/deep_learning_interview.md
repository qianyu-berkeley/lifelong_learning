# Kindergarten
## Logistic Regression

**Key Concepts**

* odds, odds ratio, and probability

    $$odds(p) = (\frac{p}{1-p})$$
    $$odds\ ratio = \frac{\frac{X_A}{X_B}}{\frac{Y_A}{Y_B}}$$
    $$relative\_risk = \frac{\frac{X_A}{X_A+Y_A}}{\frac{X_B}{X_B+Y_B}}$$
    where X is treated, Y is control, A is impacted, B is not impacted
    $$ probability = \frac{odds}{1+odds} = \frac{4}{1+4} = 0.8$$

* Distribution of logistic regression predictor and outcome variables

    $$Z = logit(P) = log(odds) = log(\frac{P}{1-P}) = \theta^Tx = \theta_0 + \theta_1$$
    $$e^Z = \frac{P}{1-P}$$
    $$P = \frac{e^Z}{1+e^Z} = \frac{1}{1+e^{-Z}}$$

* Sigmoid function (logistic function for binary classification and a neuron activation function)
  $$\sigma(x) = \frac{1}{1+e^{-\theta x}}$$
  
  *Note*: Sigmoid or softmax function output is also **logit**

  Derivative of sigmoid funtion (we can expand this to softmax) 

  $$\frac{d}{dx}\sigma(x)=\frac{e^{-x}}{(1+e^{-x})^2}$$
  or 
  $$\sigma'(x) = \sigma(x)(1-\sigma(x)) $$ 

* Logistic Regression Definition (put the above concept together)
  Hypothesis function $h_{\theta}(x)$
  Logit: $Z = \theta^Tx$
  $$h_{\theta}(x) = \frac{1}{1+e^Z} = \frac{1}{1+e^{-\theta^T x}}$$

  Decision Boundry:
  $$h_{\theta}(x) \geq 0.5  \to y = 1$$
  $$h_{\theta}(x) < 0.5  \to y = 0$$
  or
  $$\theta^T \geq 0 \to y = 1$$
  $$\theta^T < 0 \to y = 0$$

  Cost Function (Measure the goodness of our hypothesis with respect to all data samples)
  $$J(\theta) = \frac{1}{m} \sum^m_{i=1}Cost(h_\theta(x^{(i)}), y^(i))$$
  $$J(\theta) = \frac{1}{m} \sum^m_{i=1}(-y^ilog(h_\theta(x^i)) - (1-y^i)log(1-h_\theta(x^i)) )$$
  $$J(\theta) = -\frac{1}{m} \sum^m_{i=1}(y^ilog(h_\theta(x^i)) + (1-y^i)log(1-h_\theta(x^i)) )$$




## Questions and Answers
### prb-4
True or False: For a fixed number of observations in a data set, introducing more variables normally generates a model that has a better fit to the data. What may be the drawback of such a model fitting strategy?

AN: True. Overfitting

### prb-5
Define the term “odds of success” both qualitatively and formally. Give a numerical example that stresses the relation between probability and odds of an event occurring.

**AN**: 
Odds of success = probability of success / probability of failure whereas probability is freq of success / total
Using the event of rolling a dice: odds of rolling a 6 is 1/5, prob of rolling a 6 is 1/6
$$Odds(p) = (\frac{p}{1-p})$$


### prb-6

Define what is meant by the term "interaction", in the context of a logistic regression predictor variable.

**AN**: 
1. An interaction is the product of two single predictor variables implying a non-additive effect.

2. What is the simplest form of an interaction? Write its formula?
AN: The simplest interaction model includes a predictor variable formed by multiplying two ordinary predictors. Let us assume two variables X and Z. Then, the logistic regression model that employs the simplest form of interaction follows: 

$$\beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_1X_2$$

$\beta_3$ is the coefficient of the interaction term

3. What statistical tests can be used to attest the significance of an interaction term?

For testing the contribution of an interaction, two principal methods are commonly employed; the Wald chi-squared test or a likelihood ratio test between the model with and without the interaction term.


### prb-7

True or False: In machine learning terminology, unsupervised learning refers to the mapping of input covariates to a target response variable that is attempted at being predicted when the labels are known

**AN**: 
false, labels are unknown for unsupervised learning, the description describe the supervised learning model


### prob-8

Complete the following sentence: In the case of logistic regression, the response variable is the log of the odds of being classified in [...]

**AN**:
in a group of binary or multi-class responses


### prb-9
Describe how in a logistic regression model, a transformation to the response variable is applied to yield a probability distribution. Why is it considered a more informative representation of the response?

**AN**:
When a transformation to the response variable is applied, it yields a probability distribution over the output classes, which is bounded between 0 and 1; this transformation can be employed in several ways, e.g., a softmax layer, the sigmoid function or classic normalization.  This representation facilitates a soft-decision by the logistic regression model, which permits construction of probability-based processes over the predictions of the model.

* Softmax is for multi-classs problem
* Sigmoid is for binary class problem

### prb-10
Minimizing the negative log likelihood also means maximizing the [...] of selecting the [...] class

**AN**: 
probability/likelihood of selecing the correct class


### prb-11

Assume the probability of an event occurring is p = 0.1.

1. What are the odds of the event occurring?.

    $$odds = \frac{p}{1-p} = \frac{0.1}{0.9} = 0.111$$

2. What are the log-odds of the event occurring?.
    $$log\ odds = \ln({\frac{p}{1-p}}) = \ln(0.1/0.9) = -2.197$$

3. Construct the probability of the event as a ratio that equals 0.1.

$$odds = \frac{p}{1-p}$$
$$p = \frac{odds}{1+odds} = \frac{0.11}{1+0.11} = 0.1$$


### prb-12
True or False: If the odds of success in a binary response is 4, the corresponding probability of success is 0.8.

**An**: 
True

$$ p = \frac{odds}{1+odds} = \frac{4}{1+4} = 0.8$$

### prb-13
Draw a graph of odds to probabilities, mapping the entire range of probabilities to their respective odds.

$$ odds = \frac{P}{1-P}$$

If we plot $y(x) = \frac{x}{1-x}$

* Assume x axis is prob, y axis is odds
* Wnen probability is 0, odds is 0
* When probability is 1. odds is infinity
* It is concav up and asymptote to infinity when prob at 1 

### prob-14
The logistic regression model is a subset of a broader range of machine learning models known as generalized linear models (GLMs), which also include analysis of variance (ANOVA), vanilla linear regression, etc. There are three components to a GLM; identify these three components for binary logistic regression.

3 components of GLM: 
* Random component: refers to the probability distribution of the response variable (Y)
* Systematic component: describes the explanatory variables
* Link function: specifies the link between random and systematic components

For binary logistic regression:
* The Random cmoponent is binormial distribution
* Systematic component is $\sum{\theta_i x_i}$
* Link function: how the expected value of the response relates to the linear predictor of explanatory variables. 


### prb-15
Let us consider the logit transformation, i.e., log-odds. Assume a scenario in which the logit forms the linear decision boundary:

$$\log(\frac{Pr(Y=1|X)}{Pr(Y=0|X)}) = \theta_0 + \theta^TX $$

for a given vector of systematic components X and predictor variables $\theta$. Write the mathematical expression for the hyperplane that describes the decision boundary

**AN**: 
Decision boundry of logistic regression gives:
$$h(\theta) = g(\theta^T x) = \frac{1}{1+e^{-\theta^T x}} = 0.5$$
Therefore
$$ e^{-\theta^T x} = 1$$
or
$$\theta_0 + \theta^TX = 0 $$



### prb-16
True or False: The logit function and the natural logistic (sigmoid) function are inverses of each other.

**AN**: True

Proof:

We know logit function is defined as:
$$Z = logit(P) = log(\frac{P}{1-P})$$

taking the inverse of logit:
$$e^Z = \frac{P}{1-P}$$
we get
$$P = \frac{e^Z}{1+e^Z}$$
or 
$$P = \frac{1}{1+e^{-Z}}$$
Which is the sigmoid function

### prb-17
Compute the derivative of the natural sigmoid function: $\sigma(x) = \frac{1}{1+e^{-x}} \in (0, 1)$

**AN:**

$$\frac{d}{dx}\sigma(x)=\frac{e^{-x}}{(1+e^{-x})^2} \in (\frac{1}{4}, \frac{e}{(1+e)^2})$$

The weakness of signmoid function as activiation function is the when the input arguement is very large or very small the signmoid function is very flat, its derivative becomes very small therefore the training becomes very slow

### prb-18
Remember that in logistic regression, the hypothesis function for some parameter vector $\beta$ and measurement vector x is defined as:

$$ h_{\beta}(x) = g(\beta^Tx) = \frac{1}{1 + e^{-\beta^Tx}} = P(y=1|x; \beta)$$

where y holds the hypothesis value.  Suppose the coefficients of a logistic regression model with independent variables are as
follows: $\beta_0=-1.5, \beta_1 = 3, \beta_2 = -0.5$, As a result, the logit equestion becomes:

**AN:**

$$logit = \beta_0 + \beta_1x_1 + \beta_2x_2$$

1. What is the value of the logit for this observation?
$$logit = -1.5 + 3 * 1 - 0.5 * 5 = -1$$
2. What is the value of the odds for this observation?
$$odds = e^{logit} = e^{-1} = \frac{1}{e}$$
3. What is the value of $P(y = 1)$ for this observation?
$$p = \frac{e^{-1}}{1+e^{-1}} = \frac{1}{1+e}$$


### prb-19
Proton therapy questions

Tumour eradication table:

|Cancer Type | Yes | No
| -----------| ----| ---
| Breast | 560 | 260 | 
| lung | 69 | 36 | 

1. What is the explanatory variable and what is the response variable?

explanatory variable (X): cancer type
response variable (Y): tumour eradication

2. Explain the use of relative risk and odds ratio for measuring association.

$$odds\_ratio = \frac{\frac{Y_1}{Y_2}}{\frac{N_1}{N_2}}$$
$$relative\_risk = \frac{\frac{Y_1}{Y_1+N_1}}{\frac{Y_2}{Y_2+N_2}}$$

Relative risk (RR) is the ratio of risk of an event in one group (e.g., exposed group) versus the risk of the event in the other group (e.g., non-exposed group). The odds ratio (OR) is the ratio of odds of an event in one group versus the odds of the event in the other group.

$$odds\_ratio = \frac{\frac{560}{69}}{\frac{260}{36}} = 1.123$$
$$relative\_risk = \frac{\frac{560}{820}}{\frac{69}{105}} = 1.039$$

3. Are the two variables positively or negatively associated? Find the direction and strength of the association using both relative risk and odds ratio.

Yes, they are positively correlated
The odds ratio is larger than one, indicating that the odds for a breast cancer is more than the odds for a lung cancer to be eradicated. 

4. Compute a 95% confidence interval (CI) for the measure of association.
To test association, we will use chi-square test

|Cancer Type | Yes | No | total
| -----------| ----| ---|------
| Breast | 560 | 260 | 820
| lung | 69 | 36 | 105
| total | 629 | 296 | 925

Breast OR: (560/820)/(1-(560/820)) = 2.15
lung OR: (69/105)/(1-(69/105)) = 1.91
OR = 2.15/1.91 = 1.13

95% CI of odd ratio
$$\log(OR)\pm1.96\sqrt{\frac{1}{a}+\frac{1}{b}+\frac{1}{c}+\frac{1}{d}})$$ 
$$95\%\ CI\ log\ odds = \log(1.123)\pm1.96\sqrt{\frac{1}{560}+\frac{1}{260}+\frac{1}{69}+\frac{1}{36}})$$ 
$$95\%\ CI\ odds = e^{log(1.123)\pm1.96\sqrt{\frac{1}{560}+\frac{1}{260}+\frac{1}{69}+\frac{1}{36}})}$$ 

$$CI = (0.810, 1.909)$$

5. Interpret the results and explain their significance.
The CI (0.810, 1.909) contains 1, which indicates that the true odds ratio is not significantly different from 1 and there is not enough evidence that tumour eradication is dependent on cancer type.

### prb-20

1. Estimate the probability that, given a patient who undergoes the treatment for 40 milliseconds and who is presented with a tumour sized 3.5 centimetres, the system eradicates the tumor.

$e^{(-6 + 0.05 * 40 + 1*3.5)} = 0.61$
$p = \frac{0.61}{1+0.61} = 0.38$


2. How many milliseconds the patient in part (a) would need to be radiated with to have exactly a 50% chance of eradicating the tumor?

50 millisecond

### prb-21

1. Using X1 and X2, express the odds of a patient having a migraine for a second time.  

    $$P = \frac{1}{1+e^{-\beta^Tx}}$$ 
    where $$\beta^Tx = -6.36 - 1.02x_1 + 0.12x_2$$
    $$odds = e^{-6.36 - 1.02x_1 + 0.12x_2}$$

2. Calculate the probability of a second migraine for a patient that has at least four amalgams and drank 100 cups per month?

    We plug in 1 for $x_1$ and 100 for $x_2$, we get p = 0.99 or 99%

3. For users that have at least four amalgams, is high coffee intake associated with an increased probability of a second migraine?

    Yes, the coefficient for X2 (0.119) is a positive number and P-value is 0.0304< 0.05

4. Is there statistical evidence that having more than four amalgams is directly associated with a reduction in the probability of a second migraine?

    No, since the P-value for the coefficient is 0.3818 > 0.05 and is not statistically significant

### prb-22

1. Estimate the probability of improvement when the count of gum bacteria of a patient is 33.

    $$P = \frac{1}{1+e^{-\beta^Tx}}$$
    where $$\beta^Tx = -4.88 + 0.0258x$$

    $$p = \frac{1}{1+e^{-(-4.88 + 0.0258*33)}} = 0.017$$ 

2. Find out the gum bacteria count at which the estimated probability of improvement is 0.5.

    $$P = \frac{1}{1+e^{-4.88+0.0258x}} = 0.5$$
    $$e^{-(-4.88+0.0258x)} = 1$$
    $$-4.88+0.0258x = 0$$
    $$x = 189$$
    
    The bacteria count is 189

3. Find out the estimated odds ratio of improvement for an increase of 1 in the total gum bacteria count.

    $$odds\ ratio = odds_{(x+1)} / odds_{(x)}$$
    $$log(odds\ ratio) = log(odds_{(x+1)}) - log(odds_{(x)})$$
    $$log(odds\ ratio) = -4.8792 + 0.0258(x+1) - (-4.8792 + 0.0258x) = 0.0258$$
    $$odds\ ratio = e^{0.0258} = 1.0261$$

4. Obtain a 99% confidence interval for the true odds ratio of improvement increase of 1 in the total gum bacteria count. Remember that the most common confidence levels are 90%, 95%, 99%, and 99.9%. Table 9.1 lists the z values for these levels.

$$99\%\ CI = 0.0258 \pm 2.576 \times 0.0194$$
$$99\%\ True\ CI = e^{0.0258 \pm 2.576 \times 0.0194}$$



### prb-23

1. Find the sample odd ratio

$$odd\ ratio = \frac{\frac{60}{130}}{\frac{6833}{6778}} = 0.458 $$ 

2. Find the sample log-odd ratio

$$log\ odds\ ratio = log(0.458) $$

3. Compute a 95% confidence interval (z0.95 = 1.645; z0.975 = 1.96) for the true log odds ratio and true odds ratio.

99% CI of odd ratio = $e^{(-0.783\pm1.96\sqrt{\frac{1}{60}+\frac{1}{130}+\frac{1}{6833}+\frac{1}{6778}})}$ 


### prb-24

Entropy loss of a single binary outcome with probability p

$$H(p) = -plog(p) - (1-p)log(1-p)$$

1. At what p does H(p) attain its maximum value?
when p = 0.5, H(p) = 0.693

2. What is the relationship between the entropy H(p) and the logit function, given p?

$$\frac{dH(p)}{dp} = -logit(p)$$


### prb-25

```cpp
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

std::vector<double> theta {-6,0.05,1.0};
double sigmoid(double x) {
  double tmp =1.0 / (1.0 + exp(-x));
  std::cout << "prob=" << tmp<<std::endl;
  return tmp;
}

double hypothesis(std::vector<double> x){
  double z;
  z=std::inner_product(std::begin(x), std::end(x), std::begin(theta), 0.0);
  std::cout << "inner_product=" << z <<std::endl;
  return sigmoid(z);
}

int classify(std::vector<double> x){
  int hypo=hypothesis(x) > 0.5f;
  std::cout << "hypo=" << hypo<<std::endl;
  return hypo;
}
int main() {
  std::vector<double> x1 {1,40,3.5};
  classify(x1);
}
```

1. Explain the purpose of line 10, i.e., inner_product

Calculate the logit function, or probability of binary class

2. Explain the purpose of line 15, i.e., hypo(x) > 0.5f

make binary classification of prob > 0.5 return true otherwise return false

3. What does $\theta$ stand for in line 2?
coefficient of logistic regression

4. Compile and run the code, you can use: https://repl.it/languages/cpp11 to evaluate the code.  What is the output?

inner_product=-0.5
prob=0.377541
hypo=0


### prob-26

```python
import torch
import torch.nn as nn

lin = nn.Linear(5, 7)
data = (torch.randn(3, 5))
print(lin(data).shape)
```

shape is (3, 7)


### prob-27
```python
from scipy.special import expit
import numpy as np
import math

def Func001(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def Func002(x):
    return 1 / (1 + math.exp(-x))

def Func003(x):
    return x * (1-x)
```

Func001 is a softmax function
Func002 is a sigmoid function
Func003 is the derivative of a sigmoid function

Note: $$\sigma'(x) = \sigma(x)(1-\sigma(x))$$

### prob-28
```python
from scipy.special import expit
import numpy as np
import math

def Func006(y_hat, y):
    if y == 1:
        return -np.log(y_hat)
    else:
        return -np.log(1 - y_hat)
```

What important concept in machinelearning does it implement?

AN: It implement the binary cross-entropy function (negative log-loss)

### prob-29

```python
from scipy.special import expit
import numpy as np
import math

def Ver001(x):
    return 1 / (1 + math.exp(-x))

def Ver002(x):
    return 1 / (1 + (np.exp(-x)))

WHO_AM_I = 709
def Ver003(x):
    return 1 / (1 + np.exp(-(np.clip(x, -WHO_AM_I, None))))
```

1. Which mathematical function do these methods implement?
The sigmoid objective function of logistic regression (probability)


2. What is significant about the number 709 in line 11?
exceeding pyhton floating point number boundry

3. Given a choice, which method would you use?
Ver003 is the best to ensure numerical stability

## Probabilistic Programming and Bayesian DL

**Key Concepts**
* Probability Theory
  * Likelihood and Log Likelihood

    with Laws of total probability:

    $$P(A) = P(A \cap B) + P(A \cap B^c)$$
    $$P(A) = P(A | B)P(B) + P(A | B^c)P(B^c)$$

  * Bayes Theorem:

    $$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A^c)P(A^c)}$$
    $$ P(B|A) = \frac{P(A|B)P(B)}{P(A)} = \frac{P(A|B)P(B)}{P(A|B)P(B) + P(A|B^c)P(B^c)}$$


    **Likelihood**
    Many probability distributions have unknown parameters; We estimate these unknowns using sample data. The Likelihood function gives us an idea of how well the data summarizes these parameters (parameters for PDF) or how well a parameter explain the observed data.

    Likelihood is different from PDF where PDF is a function random variable X treating parameters such as $\theta$ as constants. A likelihood function, on the other hand, takes the data set as a given, and represents the likeliness of different parameters for your distribution where it is a function of $\theta$ thus they are not constant.

    Likelihood function describes the join probability of the observed data as a function of parameters of the chosen statistical model. 

    $$L(\theta | x) = P(x | \theta)$$

    Likelihoods are a key part of Bayesian inference. We also use likelihoods to generate estimators; we almost always want the maximum likelihood estimator.

    Maximum Likelihood Estimator:
    $$\hat{\theta} = argmax_{\theta \in \Theta}(L(\theta|X))$$

    **log likelihood** 
    log likelihood function is usually preferred to likelihood for a few reasons:
    * The log likelihood function in maximum likelihood estimations is usually computationally simpler
    * Likelihoods are often tiny numbers (or large products) which makes them difficult to graph. Taking the natural (base e) logarithm results in a better graph with large sums instead of products
    * The log likelihood function is usually (not always!) easier to optimize.
* Binomial Distribution

 $$X \sim Binomial(n, p)$$
 probability Mass Function (PMF)
 $$P(X=k) = \binom nk p^k (1-p)^{n-k}$$
 Expectation
 $$E(X) = np$$
 Variance
 $$VAR(X) = npq = np(1-p)$$

* Bernoulli distribution (one independent trial from Binomail distribution)
 x is a bernoulli distribution with parameter p and x has exactly 2 possibility $\{a, b\}$
 PMF of Bernoulli distribution is defined as:
 $$p(x) = \begin{cases} p : x = a\\ 1-p : x = b \\ 0 x \notin {a, b} \end{cases}$$
 Expectation:
 $$E(x) = p$$
 Variance:
 $$VAR(X) = p(1-p)$$
 if $x \in \{0, 1\}:$
 $$p(x) = p^x(1-p)^{(1-x)}$$

* Binomial likelihood
 $$L(p \mid n, k) = \binom{n}{p}p^k(1-p)^{n-k}$$

  * The likelihood function is **NOT** a probablility function
  * It is a positive function where $0 \leq p \leq 1$
  * Left-hand size: the likelihood of the parameter $p$ given n and k
  * The right and side appears to be the same as the PMF of Binomail distribution but the difference is the condition of the left hand side
  * The **probability function** returns the probability of the data with given sample size ($n$) and the parameters whereas the **likelihood function** gives the relative likelihoods for different values of the parameter ($p$) given the sample size ($n$) and the data ($k$)
  * log-likelihood of often more useful since it turn things into addition
    $$\ln(L) = \ln(L(p \mid n, k)) = ln{\binom{n}{k}} + k\cdot \ln(p) + (n-k)\cdot \ln(1-p)$$
  * Example: 
    * A probability problem: given an unfair coin with p, if we flip n times,  what is the probabiliy of getting k heads
    * A likelihood problem: given an unfair coin with p, what is the likelihood function of flip n times and getting k head. We will have plot with p as x-axis and likelihood function as y-axis

* Beta Distribution
* Bayesian Statistics
* Probabilistic library (PyMc3, Stan)

## Questions and Answers

### Expectation and Variance

### PRB-30
Define what is meant by Bernoulli trial

AN: 
Bernoulli trial also named as binomial trial is a random experiment with exactly 2 possible outcomes (success or failure), in which the probability of success is the same every time the experiment is conducted.

### PRB-31
The binomial distribution is often used to model the probability that k out of a group of n objects bare a specific characteristic. Define what is meant by a binomial random variable X.

AN: 
A binomial random variable counts how often a particular event occurs in a fixed number of tries or trials:
* A fix number of trials
* binary outcome of each trial
* the probability of occurance (succcessful / trial) is the same of each trial
* Trials are independent of one another

An example is the probability of whether a characteriztic exists or not out of n objects.

### PRB-32
What does the following shorthand stand for?

$$X \sim Binomial(n, p)$$

AN:
This is shorthand stand for a binomial random variable with sample size of n and probabiliy of an event of interest occurs at a trial is p

### PRB-33
Find the probability mass function (PMF) of the following random variable:

$$X \sim Binomial(n, p)$$

AN:

$$P(X=k) = \binom nk p^k (1-p)^{n-k}$$

Describe the probability of getting k successes in n independent bernoulli trials.


### PRB-34

1. Define what is meant by (mathematical) expectation.

$$E(X) = \sum_{i=0}^k {x_ip_i}$$

where x is random variable and p is probability

2. Define what is meant by variance.

$$Var(X) = E((x - E(X))^2)$$ 
$$Var(X) = E(X^2) - (E(X))^2$$ 

3. Derive the expectation and variance of a the binomial random variable $X \sim Binomial(n, p)$ in terms of p and n.

$$E(X) = np$$

We derive from the fact that Binomial distribution os consiste of n independent Bernoulli trails and each Bernoulli trail has random variable Y with parameter p (i.e. $E(Y) = p$)


$$E(X) = E(\sum_{i=1}^n Y_i) = \sum_{i=1}^nE(Y_i)$$
$$ = \sum_{i=1}^n p = np$$

$$Var(X) = np(1-p)$$

We derive as n times the variance of bernoulli distrubtion

$$Var(X)=  n*Var(Y) = n(p(1-p))$$


### PRB-35
Proton therapy (PT) is a widely adopted form of treatment for many types of cancer.  A PT device which was not properly calibrated is used to treat a patient with pancreatic cancer (Fig. 3.1). As a result, a PT beam randomly shoots 200 particles independently and correctly hits cancerous cells with a probability of 0.1.

1. Find the statistical distribution of the number of correct hits on cancerous cells in the described experiment. What are the expectation and variance of the corresponding random variable?

$$X \sim Binomial(200, 0.1)$$
$$E(x) = 200*0.1 = 20$$
$$Var(x) = 200*0.1*0.9 = 18$$

2. A radiologist using the device claims he was able to hit exactly 60 cancerous cells.  How likely is it that he is wrong?

$$ \sigma = \sqrt{18} = 4.2$$
$$ \frac{60 - 20}{4.2} \gt 3\sigma$$

or

$$P(X = 60; n=200, p = 0.1) = \binom{200}{60}0.1^{60}(1-0.1)^{200-60} = 2.7 \times e^{-15}$$

He is likely wrong is larger than 99% or his probability of sucess is extremely low

### PRB-36
Given two events A and B in probability space H, which occur with probabilities P(A) and P(B), respectively:

1. Define the conditional probability of A given B. Mind singular cases.

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

2. Annotate each part of the conditional probability formula

Probability of A given B is equal to probability of A and B divided by probability of B

3. Draw an instance of Venn diagram, depicting the intersection of the events A and B.  Assume that $A \cap B = H$.

$P(A \cup B) = P(A) + P(B) - H$

### PRB-37
Bayesian inference amalgamates data information in the likelihood function with known prior information. This is done by conditioning the prior on the likelihood using the Bayes formulae. Assume two events A and B in probability space H, which occur with probabilities P(A) and P(B), respectively. Given that $A \cup B = H$, state the Bayes formulae for this case, interpret its components and annotate them.

Laws of total probability:

$$P(A) = P(A \cap B) + P(A \cap B^c)$$
$$P(A) = P(A | B)P(B) + P(A | B^c)P(B^c)$$

Bayes Theorem:

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A^c)P(A^c)}$$
$$ P(B|A) = \frac{P(A|B)P(B)}{P(A)} = \frac{P(A|B)P(B)}{P(A|B)P(B) + P(A|B^c)P(B^c)}$$

A can be interpret as the following: 
* if A is hypothesis or prior information and B is evidence or data
* $P(A \mid B)$ is the posterior probability of an event
* $P(A)$ is the prior (information or hypothesis)
* $P(B \mid A)$ is the likelyhood the data with given (information or hypothesis)
* $P(B)$ is the normalization constant or total probability of data

### PRB-38
Define the terms likelihood and log-likelihood of a discrete random variable X given a fixed parameter of interest $\gamma$. Give a practical example of such scenario and derive its likelihood and log-likelihood.

Likelihood function of parameter $\gamma$ with a discrete random variable X is

$$L_\gamma(X = x) = P(X = x \mid \gamma)$$

log likelihood

$$ln(L_\gamma(X = x)) = ln(P(X = x \mid \gamma))$$

Since $\gamma$ is constant, it becomes the conditional probability of X given $\gamma$ or how likely is to obtain a value x when a prior information $\gamma$ is given regarding its distribution

A practical example is the probability of a patient actually getting a disease if he/she is tested positive for the disease. The $\gamma$ is a parameter describe the disease test either positive or negative but in this case it is fixed (positive) 

* $P(D_+ | T_+)$: Probability of have disease given tested positive
* $P(T_+| D_- )$: Probability of tested positive but does not have disease (False Positive)
* $P(T_-| D_+ )$: Probability of tested negative but have disease (False Negative) 
* $P(D_+)$: Probability of the population has disease (prior pobability)
* $P(T_-|D_-) = 1 - P(T_+|D_-)$
* $P(T_+|D_+) = 1 - P(T_-|D_+)$

Based on bayes theorem:
$$P(D_+ | T_+) = \frac{P(T_+|D_+)P(D_+)}{P(T_+)}$$
$$= \frac{P(T_+|D_+)P(D_+)}{P(T_+|D_+)P(D_+) + P(T_+|D_-)P(D_-)}$$

For continous distribution (such as normal distribution), we often use log-likelihood which is 
$$ ln(L(x; \mu, \sigma)) = ln(p(X=x; \mu, \sigma | \gamma))$$ 
where it follows a normal distribution, likelihood is essentially the y value with given x in a normal distribution PDF. (e.g. the likelihood of getting 0.7 for a normaly distributed data with $\mu=0.5$, $\sigma=1$)

### PRB-39
Define the term prior distribution of a likelihood parameter $\gamma$ in the continuous case.

The continuous prior distribution $f(\Gamma = \gamma)$ represents what is known about the probability of the value before the experiment has commenced. It is termed as being subjective, and therefore may vary considerably between researchers. By proceeding the previous example, $f(\Gamma = 0.8)$ holds the probability of randomly flipping a coin that yields “heads” with chance
of 80% of times.

### PRB-40
Show the relationship between the prior, posterior and likelihood probabilities.

Posterior probability refers to the conditional probability $P(\theta \mid X)$, i.e. probability of parameter $\theta$ given an event X which comes from an application of Bayes' theorem

**Elaborate**:
The essence of Bayesian analysis is to draw inference of unknown quantities or quantiles from the posterior distribution $p(\Theta = \theta \mid X = x)$ , which is traditionally derived from prior beliefs and data information. 

Bayesian statistical conclusions about chances to obtain the parameter $\Theta = \theta$ or unobserved values of random variable $X = x$, are made in terms of probability statements. These probability statements are conditional on the observed values of X, which is denoted as $p(\Theta = \theta \mid X = x)$, called posterior distributions of parameter. Bayesian analysis is a practical method for making inferences from data and prior beliefs using probability models for quantities we observe and for quantities which we wish to learn


$$P(\theta \mid X) = \frac{P(X \mid \theta)P(\theta)}{P(X)}$$

$P(\theta \mid X)$ is the posterior probability, prior is $P(\theta)$ and the likelihood is $P(X \mid \theta)$


### PRB-41
In a Bayesian context, if a first experiment is conducted, and then another experiment is followed, what does the posterior become for the next experiment

$$P(\Theta = \theta \mid X=x) = \frac{P(X \mid \theta)P(\theta)}{P(X)}$$

This is part of the well-known Bayesian updating mechanism wherein we update our knowledge to reflect the actual distribution of data that we observed. To summarize, from the perspective of Bayes Theorem, we update the prior distribution to a posterior distribution after seeing the data.

### PRB-42
What is the condition under which two events A and B are said to be statistically independent?

$$P(A \cap B) = P(A)P(B)$$
or
$$P(B \mid A) = P(B)$$

The intuitive meaning of the definition in terms of conditional probabilities is that the probability of B is not changed by knowing that A has occurred.

### Bayes Rule

### PRB-43
In an experiment conducted in the field of particle physics (Fig. 3.2), a certain particle may be in two distinct equally probable quantum states: integer spin or half-integer spin.  It is well-known that particles with integer spin are bosons, while particles with half-integer spin are fermions

A physicist is observing two such particles, while at least one of which is in a half-integer state. What is the probability that both particles are fermions?

If we Define $\gamma$ as the number of half-integer spin states

$$P(\gamma=2 \mid \gamma \geq 1) = \frac{P(\gamma = 2 \cap \gamma \geq 1)}{P(\gamma \geq 1)}$$
$$ =  \frac{P(\gamma=2)}{P(\gamma \geq 1)}$$

Because $\gamma = 2$ is a subset of $\gamma \geq 1$ 

$$ =  \frac{P(\gamma=2)}{1 - P(\gamma = 0)}$$
$$ = \frac{1/4}{1 - 1/4} = \frac{1}{3}$$

### PRB-44
During pregnancy, the Placenta Chorion Test [1] is commonly used for the diagnosis of hereditary diseases (Fig. 3.3). The test has a probability of 0.95 of being correct whether or not a hereditary disease is present

It is known that 1% of pregnancies result in hereditary diseases. Calculate the probability of a test indicating that a hereditary disease is present.

Given:
* $P(D_+) = 0.01$
* $P(D_-) = 0.99$
* $P(T_+ \mid D_+) = 0.95$
* $P(T_- \mid D_+) = 0.05$
* $P(T_- \mid D_-) = 0.95$
* $P(T_+ \mid D_-) = 0.05$

We are looking for $P(T_+)$ , using law of total probability

$$P(T_+) = P(T_+ \mid D_+)P(D_+) + P(T_+ \mid D_-)(P(D_-)$$
$$ = 0.95 \cdot 0.01 + 0.05 \cdot 0.99 = 0.059 $$

### PRB-45
The Dercum disease [3] is an extremely rare disorder of multiple painful tissue growths.  In a population in which the ratio of females to males is equal, 5% of females and 0.25% of males have the Dercum disease

A person is chosen at random and that person has the Dercum disease. Calculate the probability that the person is female.

Given:
* $P(F) = 0.5$
* $P(M) = 0.5$
* $P(D_+ \mid F) = 0.05$
* $P(D_+ \mid M) = 0.0025$

$$P(F \mid D_+) = \frac{P(D_+ \mid F)P(F)}{P(D_+)}$$
$$ = \frac{P(D_+ \mid F)P(F)}{P(D_+ \mid F)P(F) + P(D_+ \mid M)P(M)}$$
$$ = \frac{0.05 \cdot 0.5}{0.05 \cdot 0.5 + 0.0025 \cdot 0.5}$$
$$ = 0.952$$

### PRB-46

There are numerous fraudulent binary options websites scattered around the Internet, and for every site that shuts down, new ones are sprouted like mushrooms. A fraudulent AI based stock-market prediction algorithm utilized at the New York Stock Exchange, (Fig. 3.6) can correctly predict if a certain binary option [7] shifts states from 0 to 1 or the other way around, with 85% certainty

A financial engineer has created a portfolio consisting twice as many state-1 options then state-0 options. A stock option is selected at random and is determined by said algorithm to be in the state of 1. What is the probability that the prediction made by the AI is correct?

Define:
* $AI_1$: prediction the state of stock option is 1
* $AI_0$: prediction the state of stock option is 0
* $s_1$: the state of option is 1
* $s_0$: the state of option is 0

$P(AI_1 \mid s_1) = 0.85$
$P(AI_0 \mid s_0) = 0.85$

$$P(s_1 \mid AI_1) = \frac{P(AI_1 \mid s_1)P(s_1)}{P(AI_1)}$$
$$ = \frac{P(AI_1 \mid s_1)P(s_1)}{P(AI_1 \mid s_1)P(s_1) + P(AI_1 \mid s_0)P(s_0)}$$
$$ = \frac{0.85 \cdot 2/3}{0.85 \cdot 2/3 + 0.15 \cdot 1/3} = 0.9189$$

### PRB-47
In an experiment conducted by a hedge fund to determine if monkeys (Fig. 3.6) can outperform humans in selecting better stock market portfolios, 0.05 of humans and 1 out of 15 monkeys could correctly predict stock market trends correctly.

From an equally probable pool of humans and monkeys an “expert” is chosen at random.  When tested, that expert was correct in predicting the stock market shift. What is the probability that the expert is a human?

Given
* $P(correct \mid H) = 0.05$
* $P(correct \mid M) = 1/15$

$$P(H \mid correct) = \frac{P(correct \mid H)P(H)}{P(correct)}$$
$$ =\frac{P(correct \mid H)P(H)}{P(correct \mid H)P(H) + P(correct | M)P(M)}$$
$$ = \frac{0.05 \cdot 0.5}{0.05 \cdot 0.5 + 1/15 \cdot 0.5}$$
$$ = 0.428$$

### PRB-48

During the cold war, the U.S.A developed a speech to text (STT) algorithm that could theoretically detect the hidden dialects of Russian sleeper agents. These agents (Fig. 3.7), were trained to speak English in Russia and subsequently sent to the US to gather intelligence.  The FBI was able to apprehend ten such hidden Russian spies [9] and accused them of being "sleeper" agents.

The Algorithm relied on the acoustic properties of Russian pronunciation of the word (v-o-k-s-a-l) which was borrowed from English V-a-u-x-h-a-l-l. It was alleged that it is impossible for Russians to completely hide their accent and hence when a Russian would say V-a-u-x-h-a-l-l, the algorithm would yield the text "v-o-k-s-a-l". To test the algorithm at a diplomatic gathering where 20% of participants are Sleeper agents and the rest Americans, a data scientist randomly chooses a person and asks him to say V-a-u-x-h-a-l-l. A single letter is then chosen randomly from the word that was generated by the algorithm, which is observed to be an "l". What is the probability that the person is indeed a Russian sleeper agent?

Given:
* $P(Sleeper) = 0.2$
* $P(nonsleeper) = 0.8$
* $P(l \mid nonsleeper) = 1/6$
* $P(l \mid sleeper) = 1/4$

$$P(sleeper \mid l) = \frac{P(l \mid sleeper)P(sleeper)}{P(l)}$$
$$ = \frac{P(l \mid sleeper)P(sleeper)}{P(l \mid sleeper)P(sleeper) + P(l \mid nonsleeper)P(nonsleeper)}$$
$$ = \frac{0.25 \cdot 0.2}{0.25 \cdot 0.2 + 1/6 \cdot 0.8} = 0.272$$

### PRB-49
During World War II, forces on both sides of the war relied on encrypted communications.  The main encryption scheme used by the German military was an Enigma machine [5], which was employed extensively by Nazi Germany. Statistically, the Enigma machine sent the symbols X and Z Fig. (3.8) according to the following probabilities:

$$P(X) = \frac{2}{9}$$
$$P(Z) = \frac{7}{9}$$

In one incident, the German military sent encoded messages while the British army used countermeasures to deliberately tamper with the transmission. Assume that as a result of the British countermeasures, an X is erroneously received as a Z (and mutatis mutandis) with a probability 1/7 . If a recipient in the German military received a Z, what is the probability that a Z was actually transmitted by the sender?

Given
* $P(X_+) = \frac{2}{9}$
* $P(Z_+) = \frac{7}{9}$
* $P(Z_+ \mid X) = 1/7$

$$P(Z_+ \mid Z) = \frac{P(Z \mid Z_+)P(Z_+)}{P(Z)}$$
$$ = \frac{P(Z \mid Z_+)P(Z_+)}{P(Z \mid Z_+)P(Z_+) + P(Z \mid X_+)P(X_+)}$$

### Maximum Likelihood Estimatation

### PRB-50
What is likelihood function of the independent identically distributed (i.i.d) random variables:

$X_1, ..., X_n$ where $X_i \sim binomial(n, p)$, $\forall i \in [1, n]$

and where p is the parameter of interests?

$$L(p \mid n, i) = \Pi^{n}_{i=1} \binom{n}{i}p^i(1-p)^{n-i} $$


### PRB-51
How can we derive the maximum likelihood estimator (MLE) of the i.i.d samples X1, ..., Xn introduced in the last problem?

$$\hat{p} = argmax(L(p ; n, x))$$

### PRB-52

What is the relationship between the likelihood function and the log-likelihood function?

log function break multiplication to sum