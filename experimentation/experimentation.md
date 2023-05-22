# Experimentation

## Sampling

* Sampling with replacement (a.k.a **Bootstrapping**): Each time the sampled values are independent. Practicially, it means that each sampling has no impact to each other, thus, the covariance between samples are 0
* Sampling without replacement: each sample values are not independent. The covariance between samples are not zero.
* For most of the exmperiment where the total population is large, we can assume covariance is close zero

## Bias

* Omitted variable bias (unobserved hetergeneity): it is the foundamental reason that we cannot use observeation to make causal inferences
* Selection bias: samples of control and treatment groups are not total randomized. Control and treatment groups are not the same intrinscially without treatment being applied
* Self-selection bias: often happens in a natrual experiment setting, undesired samples got into control or treatment group willingly due to unseen factors

## Key to field experiment is to fend omitted variable bias

* Apply Intervention
* Randomization (otherwise, we have to use natrual experiment)
* observed/unobserved factors exist in both control and treatment group

## General Experiment Design Steps

1. Formulate research questions
2. Identify variables (cause X <> effect Y)

    * Independent variable X
    * Dependent variable Y
    * Eliminate any selection bias:
      * why people in the sample come to be impacted X

3. Generate Hypothesis

    * NULL Hypothesis: treatment has no effect
    * sharp NULL: the treatment has no effect at individual sample (very rare in A/B testing)

4. Design experiment

    * What is the Treatment
    * What is the Metrics

5. Statistical Analysis of results


## Statistical Power (sensitivity)

* Statistical power of a test: the probability that a test correctly rejects the null hypothesis ($1 - \beta$ where $\beta$ is type II error), it is the same as True Positive Rate, where
  * $\alpha = P(Type\ I\ error) = P(reject\ H_0\ |\ H_0)$ or False positive
  * $\beta = P(Type\ II\ error) = P(accept\ H_0\ |\ H_0)$ or False negative
* Effect size increases => seperate of control and treatment distribution => less type II error ($\beta$) False Negative => higher power, True Positive
* Sample size increases => smaller standard error => tigher distribution => less type II error ($\beta$) False Negative => higher power, True Positive
  * Due the $\sqrt {n}$ term,, need 4 times large sample size to achieve the same effect with half of the effect size, therefore, large dosage smaller sample size is more effective than larger sample size smaller dosage
* Signal to noise ratio negatively impact statistical power (variation in the outcome)


## How do I know the effect is not reached by changes?

* Exame the sampling distribution of estimate would reach if no effect
* Perform and AA test to determine whether there is intrinsic effect without treatment just by randomizing
* P-value
  * measure the probability we see an effect without treatment and just by changes
  * e.g. p-value of 0.05 means there are 5% probability that we will see an effect without treatment just by change
  * Reflect statistical power and precision

## Regression

### Why user regression?

* Include and measure covariates can help us gain precision of our experimental treatment estimates because covariates soak up residual variance and shrink standard errors of the treatment effects (improve signal to noise ratio)
* Need to covariate balance check
  * Covariate imbalance problem: whether covarites are balanced between control and treatment group
  * Run regression with covariates. If the covariates show small, statistically insignificant difference between control and treatment group, then we can say they are balance across control and treatment group

### Rule of thumb

* $CI = x_{bar} \pm 1.96 * SE$ or roughly $2 * SE$ (use $1.64 * SE$ if 1 tail)
* The standard error determines how much variability "surrounds" a coefficient estimate. A coefficient is significant if it is non-zero. The typical rule of thumb, is that you go about two standard deviations above and below the estimate to get a 95% confidence interval for a coefficient estimate.
* if the coefficient estimate is at least two standard errors away from 0 (or in other words looking to see if the standard error is small relative to the coefficient value)
* A variable is significant if it's corresponding coefficient estimate is significantly different from zero. In other word, there is an effect (the effect is not 0)

## A/B Testing

A/B testing is one of most widely used experimentation method within the field experiment methodology

### What A/B Testing is NOT good for?

* Not idea for testing new experiences. It may result in change aversion (where user don't like changes to the norm) or novelty effect (where user see something new and test out)
* To test a new experience:
  * having a baseline
  * Time needs to to be allowed for users to adapt to the new experience (so user novelty plateaud)
* Cannot detect if you missing something
* If we face above scenarios, we can use user logs to make hypothesis and conduct other activity such as user research, focus group, etc

### Metrics to use

* Most time, we use rate to measure the usability of a site
* We use probability when measure the impact

### Binomial Distribution

* For A/B testing, events are in binomial distribution since users visit, clicks come as an event like a coin toss i.e. either they click or not click
  * Outcome are 2 types: yes or no
  * Each event are independent of the other
  * each event has an identical distribution ($p$ is the same for all)
* Binormial distribution:
  o $E(X) = p$ where $p$ is the probability of binormal distribution. (The proportion of users who clicks or visits)
  o $\sigma = \sqrt{\frac{p*(1-p)}{N}}$ where N is the number of trials

#### Confidence Interval

To measure the range where the mean is expected to fall in multiple trail of the experiment

$$ CI = \mu \pm Z*SE $$
For 95% confidence interval where Z = 1.96

#### Hypothesis Testing Setup

$$H_0: p_{t} - p_{c} = \hat{d_0} =  0$$
$$H_a: p_{t} - p_{c} = \hat{d_0} \ne 0$$

where 
* $P_{t}$ and $p_{c}$ are the treatment and control probabilities
* $\hat{d_0}$ is defined the effect size (i.e. cohen's d) 

### Comparing 2 samples (A/B testing groups)

Pooled Standard Errors:

$$SE_{pooled} = \sqrt{\hat{p_p}(1 - \hat{p_p})(\frac{1}{n_t} + \frac{1}{n_c})}$$  
where $\hat{p_p}$ is pooled probability, $p_c$, $p_t$ is probability of the control and treatment groups user clicks or visit and $n_c$, $n_t$ is the sample size of the control and treatment groups.

$$\hat{p_p} = \frac{p_tn_t + p_cn_c}{n_t+n_c}$$

We measure the effect size $d$ where

$$\hat{d} \sim N(0, SE_{pool})$$

* If $\hat{d} > 1.96 * SE_{pool}$ or $\hat{d} < -1.96 * SE_{pool}$ assuming we use 2 tails 95% confidence interval, we can reject the NULL hypothesis since it represents a statistical significant difference
* If $\hat{d} > 1.65 * SE_{pool}$ assuming we use 1 tails 95% confidence interval, we can reject the NULL hypothesis since it represents a statistical significant difference

Practical significance for a website is often around 1-2 % improvement in click through rate or user conversion rate. We should set bar lower for statistical significant than practical significant. So if the A/B test outcome satisfied practical significant, it will also meet statistical significant

### Mimimum Sample Size Calculation

A key decision is the determine the min number of samples needed to get a statistical significant results. One need to make some assumptions in:

* Statistical Power
  * Large sample size decrease $\beta$ and increase statistical power but it will not change $\alpha$
* P-value
* Effect size
* baseline rate (control group)

$$ n_{min} = \frac{2 \hat{p_{pool}}(1-p_{pool})(Z_{\beta} + Z_{\alpha/2})^2}{(p_t - p_c)^2}$$
where:
* $Z_{\beta}$: z-score corresponds to the level of statistical power assumption
* $Z_{\alpha/2}$: z-score corresponds to the level of statistical significance or confidence level

### Check variability

* Perform A/A best to evaluate variance, if you see a lot of variability in a metric, it maybe too sensitive to use it as the metric for the A/B test.
  * Compare result to what you expect (sanity check)
  * Estimate variance empirically and use your assumption about the distribution to calculate confidence
  * Directly estimate confidence interval without making any assumption of the data

### Population vs cohort

* User cohort if looking for
  * Learning effects
  * User retention
  * Test to increase user activity
  
### Analysis

1. Sanity checks

    * Check invariants of control and treatment group on whether they are indeed not vary, is the difference between control and treatment group insignificant

2. Simpson’s paradox: where the effect in aggregate may indicate one trend, and at a granular level may show an opposite trend.
3. As you increase the number of metrics, you can use a higher confidence level to overcome false positives. or use bonferroni correction although Bonferroni methods may be very conservative. 

$$\alpha_i = \frac{\alpha_{all}}{n}$$
  