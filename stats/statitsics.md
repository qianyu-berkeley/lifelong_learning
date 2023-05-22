---
title: "Statistics Concepts"
author: "Qian Yu"
date: "10/22/2020"
output: pdf_document
---

# Frequentist Statistics Basic Concepts

## Distribution

- Skewness
  - Positive: peak to right
  - Negative: peak to left
- Kurtosis:
  - Platykurtic (flat top)
  - Mesokurtic (normal top)
  - Leptokurtic (sharp top)
- Center tendency
  - mean ($\mu$)
  - median (50% tile): when distribution has large skewness, median describe center tendency better than mean
  - mode (value appeared the most)
- Disperse
  - Range
  - Variance
  - Standard deviation ($\sigma$)
  - Z-score: $\frac{X_i - \mu}{\sigma}$ describe in the unit of $\sigma$ or normalized by $\sigma$.


## Foundation Theorm of Frequentist Statistics

- **Laws of large numbers**: as a sample size grows, its mean gets closer to the average of the population. In Experiment setting, the average of the results obtained from a large number of trials should be close to the expected value and will tend to become closer to the expected value as more trials are performed
  - sample average is an unbiased and consistent estimator of population u
  - sample variance $var = \frac{var(y)}{n}$ (y is independent)
  - sample $\sigma = \frac{\sigma}{sqrt(n)}$

- **Central Limit Theorm**: The sum of a large number of independent and identically distributed random variables will be approximately normally distributed. This increase is irrespective of the shape of the actual population distribution
  - CLT allow us to perform narmal approximation of other distribution such as (Binomial, Poisson, etc)
  - CLT allow us to learn about population regardless what kind of distrbution it might be by perform random sampling and hypothesis testing
  - To measure we use CI and SE
  - Standard Error (SE) of a statistics (estimate of a parameter) is the standard deviation of its sampling distribution. Standard error defines how representative a sample of the population is likely to be
    - large SE, big variability, poor reflection
    - Small SE, small variability, good reflection
  - Standard Error of the mean (SEM): $SE = \frac {\sigma}{\sqrt(N)}$ -> It is the standard deviation of the sample means


## Hypothesis Testing

### Definition

- A test is a rule for rejecting $H_0$ (Null Hypothesis) base on the observed data and specific risk level => "Reject $H_0$ if ..."
  - $H_0$: null hypothesis $H_a$: Alternative hypothesis
  - Outcome is either reject or not reject $H_0$
- $\alpha$ = P(Type I error) = P(reject $H_0$ | $H_0$) or False positive
- $\beta$ = P(Type II error) = P(accept $H_0$ | $!H_0$)or False negative
- $\text {Statistical Power} =  1 - \beta$ or True Positive
- 1 Tail
  - $\mu < \mu_{0}$ or $\mu > \mu_{0}$  => $Z > -Z_{\alpha}$ or $Z < Z_{\alpha}$
- 2 Tails
  - $\mu != \mu_{0}$ => $|Z| > Z_{\alpha/2}$ 

![Hypothesis Test](./pics/power_table.jpg){height=40%}
