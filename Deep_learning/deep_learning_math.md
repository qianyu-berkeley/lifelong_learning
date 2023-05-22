---
title: "Deep_Learning_Math"
author: "Qian Yu"
date: "12/28/2019"
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Expected Value 
An average values of the outcomes weighted by the probability of each outcome

$$E[x] = \sum_{i=1}^n(x_iP(x_i))$$

## Pointwise Mutual Information
Measure the chance 2 outcomes tend to co-occur relative to the changes they co-occur if they are independent events

$$PMI(x, y) = \log_2{\frac{p(x, y)}{p(x)p(y)}}$$

## Entropy (Shannon entropy) 
Define how uncertain the outcome of an experiment is (the expected value $E[\log_2{p(x)}]$ for this probability distribution). 

$$Entropy(X) = H(X) = -\sum_x{p(x)\log_2{p(x)}}$$

* Binary Entropy: $p(x)\log_2{p(x)} + (1-p(x))\log_2{(1-p(x))}$. When p = 0.5, entropy is max (1.0) where as when p = 0 or p = 1 entropy is min (0.0)
* Entropy is also the average number of bits per message going across the wire given that you optimally designed your encoding scheme where $\log_2{p(x)}$ is the optimal number of bits to user for the a message.


## Cross Entropy (Machine Learning Metric: cross-entropy loss)

$$CrossEntroyp(P, Q) = -\sum_x{P(x)\log_2{Q(x)}}$$
where P is the actually probability, Q is the encoded based on limited knowledge

* The closer the distributions P and Q are to one another, the closer cross entropy will be to the entropy
* The more they differ, the bigger cross entropy will be (and the bigger the penalty for optimizing for the wrong probability distribution)
* In machine learning, we calculate $CrossEntropy(y, \hat{y})$: model predicted probability with the correct class
    - We only need to calculate one hot encoded class $P(y|x_i) = 1$
    - We can interpret average cross-entropy loss as the average number of bits needed to explain the test set labels


## KL Divergence

The difference (the size of the penalty) for using the wrong distribution to optimize the encoding (Q from P)

$$ D_{KL}(P||Q) = CrossEntropy(P, Q) - H(P)$$

* In machine learning, the KL divergence measures the "avoidable" error. 
* The unavoidable error is the bayes error rate for the underlying task


## Logistic Regression, Affine (Linear) Layer, Activation Layer

* Affine layer (If b = 0, affine becomes linear)

$$Z^{(i)}=W^Tx^{(i)} + b$$
* Activation layer

$$\hat{y}^{(i)} = a^{(i)} = sigmoid(Z^{(i)}) = \frac{1}{1+e^{-Z^{(i)}}}$$
* Objective Function

$$L(a^{(i)}, y^{(i)}) = L(\hat{y}^{(i)}, y^{(i)}) = -y^{(i)}log(a^{(i)}) - (1 - y^{(i)})log(1-a^{(i)})$$
$$= -y^{(i)}\log{\hat{y}^{(i)}}-(1-y^{(i)})\log(1-\hat{y}^{(i)})$$

$$\text{if y = 1: }L(\hat{y}, y) = -\log{\hat{y}}$$
$$\text{if y = 0: }L(\hat{y}, y) = -\log(1-\hat{y})$$ 

* Cost Function

$$J = \frac{1}{m}\sum^{m}_{i-1}{L(a^{(i)}, y^{(i)})} = -\frac{1}{m}\sum^{m}_{i-1}{y^{(i)}\log{a^{(i)}}+(1-y^{(i)})\log(1-a^{(i)})}$$
$$= -\frac{1}{m}\sum^{m}_{i-1}{y^{(i)}\log{\hat{y}^{(i)}}+(1-y^{(i)})\log(1-\hat{y}^{(i)})}$$

## Softmax

Generalized Logistic regression (sigmoid) which is a subset of softmax where the number of class = 2

* Affine layer (If b = 0, affine becomes linear)

$$Z^{(i)}=W^Tx^{(i)} + b$$
$$\hbox{softmax(x)}_{i} = \frac{e^{x_{i}}}{e^{x_{0}} + e^{x_{1}} + \cdots + e^{x_{n-1}}}$$
$$\hbox{softmax(x)}_{i} = \frac{e^{x_{i}}}{\sum_{0 \leq j \leq n-1} e^{x_{j}}}$$ 
## Cross entropy loss for some target x and prediction $p(x)$ is given by:
$$ -\sum x\, \log p(x) $$
where p(x) is calculated using log of softmax. But since our $x$s are 1-hot encoded, this can be rewritten as $-\log(p_{i})$ where i is the index of the desired target. Cross entropy loss is also called negative log likelyhood (NLL)



