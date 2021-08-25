[TOC]

```html
**Important Note**:

Almost all the contents (text, images) are came from these great books and
online resources:

* Statistics, by David Freeman, Robert Pisani, and Roger Perves

* 统计学, David Freeman $et.al$ 著，魏宗舒 等译，中国统计出版社

* 机器学习, 周志华 著 (大名鼎鼎的‘西瓜书’)

* An Introduction to Statistical Learning, by Gareth James, Daniela Witten,
    Trevor Hastie, and Robert Tibshirani

* Deep Learning, a.k.a, the flower book, by Ian Goodfellow, Yoshua Bengio, and
    Aaron Courville

* Introduction to Machine Learning, Barnabas Poczos, Aarti Singh, CMU-10701

* Bayesian Methods, Nicholas Ruozzi, UT-DALLAS
```

<h1 style="text-align: center;">Linear Regression</h1>



## Part-0: Regression 101

> $\text{You've got to draw the line somewhere.}$

### Introduction

The regression method describes how one variable depends on another. For example, take height and weight. Naturally, the taller men weighed more. How much of an increase in weight is associated with a unit increase in height? To get started, look at the scatter diagram (figure 1 on below). Height is plotted on the horizontal axis, and weight on the vertical. The summary statistics are

​		$average \ height \approx 70 \ inches,\qquad SD \approx 3 \ inches$

​		$average \ weight \approx 70 \ inches, \qquad SD \approx 45 \ pounds, \qquad r \approx 0.40$

The scales on the vertical and horizontal axes have been chosen so that one SD of height and one SD of weight cover the same distance on the page. This makes the SD line (dashed) rise at 45 degreed across the page. There is a fair amount of scatter around the line: $r$ is only 0.40.

![regression_line](./images/stats_weight_height.png)

Figure 1. Scatter diagram. Each point shows the height and weight for one of 471 men age 18-24 in a dataset. The vertical strip represents men who are about one SD above average in height. Those who are also one SD above average in weight would be plotted along the dashed SD line. Most of the men in the strip are below the SD line: they are only part of an SD above average in weight. The **solid** regression line estimates average weight at each height.

The vertical strip in figure 1 shows the men who were one SD above average in height were quite a bit less than one SD above average in weight. This is where the correlation of 0.40 comes in. Associated with an increase of one SD in height there is an increase of **only 0.40** SDs in weight, on the average.

To be more specific, take the men who are one SD above average in height:
$$
average \ height + SD \ of \ height = 70 \ in + 3 \ in = 73 \ in
$$
Their average weight will be above the overall average by $0.40 \times 45 \ lb = 18 \ lb$.

So, the average weight of these men is around
$$
average \ weight + 0.40 \times (SD \ of \ weight) = 180 \ lb + 18 \ lb = 198 \ lb
$$
The point (73 inches, 198 pounds) is marked by a cross in figure 1 ( and the points that are 2SD above (76 in, 216 lb) and below (64 in, 144 lb) the average of height as well). All the points (height, estimate for average weight) fall on the solid line shown in figure 1. This is the *regression line*. The line goes through the point of averages: men of average height should also be of average weight.

<p style="text-align:center;color:blue;">
    The regression line for y on x estimates the average value for y corresponding to each value of x.
</p>

Along the regression line, associated with each increase of one SD in height there is an increase of only 0.40 SDs in weight. Remember where the 0.40 comes from. It is the correlation between height and weight. NOTE that: Two different SDs are involved here: the SD of $x$, to gauge change in $x$; and the SD of $y$, to gauge changes in $y$.

This way of using the correlation coefficient to estimate the average value of $y$ for each value of $x$ is called the **regression method**. The method can be stated as follows.
$$
\fbox{Associated with each increase of one SD in x there is an increase of only r SDs in y, on the average.}
$$



> Correlation: Like father, like son.
>
> > If there is a strong association between two variables, then knowing one helps a lot in predicting the other. But when there is a weak association, information about one variable does not help much in guessing the other.
>
> The correlation coefficient is a measure of linear association, or clustering around a line. The relationship between two variables can be summarized by
>
> - the average of the $x$-values, the SD of the $x$-values,
> - the average of the $y$-values, the SD of the $y$-values,
> - the correlation coefficient $r$.
>
> Computing the correlation coefficient
>
> Here is the procedure for computing the correlation coefficient.
> $$
> \boxed{\text{Convert each variable to standard units.} \\
> \text{The average of the products gives the correlation coefficient.}}
> $$
>
> $$
> \fbox{Convert each variable to standard units.
> The average of the products gives the correlation coefficient.}
> $$
>
> Recall that "Convert each variable to standard units" means **standardization**. Let $x = \{1, 3, 4, 5, 7 \}, y = \{5, 9, 7, 1, 13 \} $ be vectors of variables, the mean is given by
> $$
> \mu = \frac{1}{|a|} \sum_i a_i
> $$
> And the SD is the "r.m.s size of the deviation from the average", can be given by
> $$
> \sigma = \sqrt{\frac{1}{|a|} \sum_i (a_i - \mu)^2}
> $$
> where $|a|$ is the number of data points, $a_i$ is the $i$-th data point in the data set.

> Table 1. Computing $r$.
>
> |  x   |  y   | x in standard units | y in standard units | Product |
> | :--: | :--: | :-----------------: | :-----------------: | :-----: |
> |  1   |  5   |        -1.5         |        -0.5         |  0.75   |
> |  3   |  9   |        -0.5         |         0.5         |  -0.25  |
> |  4   |  7   |         0.0         |         0.0         |  0.00   |
> |  5   |  1   |         0.5         |        -1.5         |  -0.75  |
> |  7   |  13  |         1.5         |         1.5         |  2.25   |
>
> $$
> \begin{eqnarray}
> r
> &=& \text{average  of [(x in standard units) times (x in standard units)]} \\
> \\
> &=& \frac{0.75 - 0.25 + 0.00 -0.75 + 2.25}{5} = 0.40
> \end{eqnarray}
> $$
>
> This complete the solution.

### Slope and Intercept

Does education pay? Figure 1 shows the relationship between income and education, for a sample of 562 men age 25-29 in 2005. The summary statistics are
$$
\begin{eqnarray}
\text{average education} &\approx& 12.5\ \text{years}, &\qquad& SD \approx 3\ \text{years} \\
\text{average income} &\approx& \$30000,\ &\qquad& SD \approx \$24000, \qquad r \approx 0.25
\end{eqnarray}
$$
The regression estimates for average income at each educational level fall along the regression line shown in the figure. The line slopes up, showing that on the average, income does go up with education.

![Do education pay](./images/stats_education_income.png)

Any line can be described in terms of its slope and intercept. The y-intercept is the height of the line when $x$ is $0$. And the slope is the rate at which $y$ increases, per unit increase in $x$. Slope and intercept are illustrated in figure 2.

![slopeNintercept](./images/stats_slope_intercept.png)

**How do you get the slope of the regression line?** Take the income-education example. Associated with an increase of one SD in education, there is an increase of $r$ SDs in income. On this basis, 3 extra years (one SD) of education are worth an extra $r \times SD = 0.25 \times \$24000 = \$6000$ of income, on the average. So each extra year in worth $\$6000 / 3 = \$2000$. The slope of the regression line is $\$2000$ per year.

![slopeNintercept2](./images/stats_slope_intercept2.png)

The intercept of the regression line is the height when $x = 0$, corresponding to men with $0$ years of education. There men are 12.5 years below average in education. Each year costs $\$2000$ -- that is what the slope says. A man with no education should have an income which is below average by
$$
12.5\ \text{years} \times \$2000\ \text{per year} = \$25000.
$$
His income should be $\$30000 - \$25000 = \$5000$. That is the intercept (figure 3): the predicted value of $y$ when $x = 0$.

---

<p style="text-align:justify;color:blue;">
    Associated with a unit increase in x there is some average change in y. The slope of the regression line estimates this change. The formula for the slope is
</p>

$$
{r \times SD\ \text{of y} \over SD\ \text{of x}}
$$

<p style="text-align:justify;color:blue;">
    The intercept of the regression line is just the predicted value for y when x is 0.
</p>

---

The equation of a line can be written in terms of the slope and intercept:
$$
y = \text{slope} \times x + \text{intercept},
$$
which is called the *regression equation*. There is nothing new here. The regression equation is just a way of predicting $y$ from $x$ by the regression method. 

The regression line becomes unreliable when you are far from the center of the data, so a *negative* intercept is not too disturbing (when the calculation results in some negative value which may seen absurd).

<p style="text-align:justify;color:blue;">
    If you run an observational study, the regression line only describes the data that you see. The line cannot be relied on for predicting the results of interventions.
</p>



### The Least Squares

Sometimes the points on a scatter diagram seem to be following a line. The problem discussed in this section is **how to find the line which best fits the points**. Usually, this involves a compromise: moving the line closer to some points will increase it distance from others. To resolve the conflict, two steps are necessary.

- First, define an average distance from the line to all the points.
- Second, move the line around until this average distance is as small as possible.

To be more specific, suppose the line will be used to predict $y$ from $x$. Then the error made at each point is the vertical distance from the point to the line (a.k.a, the **residual**, means the difference between the $i$th observed and the $i$th response that is predicted by linear model). In statistics, the usual way to define the average distance is by taking the root-mean-square of the errors. This measure of average distance is called the *r.m.s error of the line*. (It was first proposed by Gauss)

The second problem, how to move the line around to minimize the r.m.s error, was also solved by Gauss:

---

<p style="text-align:center;color:blue;">
    Among all lines, the one that makes the smallest r.m.s error in predicting y form x is the regression line.
</p>

---

> Recall that:
>
> The r.m.s error for regression says how far typical points are above or below the regression line.
> $$
> r.m.s\ error = \sqrt{\frac{1}{n} \sum^n_i (y_i - \hat{y_i})^2}
> $$
> where $n$ is the number of data points, $y_i$ the $i$-th actual value, $\hat{y_i}$ the corresponding predicted value.
>
> And the r.m.s error for the regression line of $y$ on $x$ can also be figured as
> $$
> \sqrt{1 - r^2} \times SD_y
> $$
> where $r$ is the correlation coefficient[^1] between $x$ and $y$.

For this reason, the regression line is often called *least squares line*: the errors are squared to compute the r.m.s error, and the regression line makes the r.m.s error as small as possible.

In other words, the least squares approaches choose $\beta_0$ (the intercept) and $\beta_1$ (the slope) to minimize the *residual sum of squares* (RSS) which is defined as
$$
RSS = e_1^2 + e_2^2 + \cdots + e_n^2 = \sum^n_{i=1} (y_i - \hat{y_i})^2
$$
where $e = y_i - \hat{y_i}$ is called the **residual**. Obviously, the r.m.s error is the root of the mean of RSS.

Linear regression is a very simple approach for supervised learning. In particular, linear regression is a useful tool for predicting a quantitative response. Many fancy statistical learning approaches can be seen as generalizations or extensions of linear regression.

$\color{Green}{\text{Example}}$

According to Hooke's law, the amount of stretch is proportional to the weight $x$. The new length of the spring is
$$
y = mx + b.
$$
In this equation, $m \in \R$ and $b \in \R$ are constants which depend on the spring. Their values are unknown, and have to be estimated using **experimental data**.

<center>Table 1. Data on Hooke's law.</center>

| Weight (kg) | Length (cm) |
| ----------- | ----------- |
| 0           | 439.00      |
| 2           | 439.12      |
| 4           | 439.21      |
| 6           | 439.31      |
| 8           | 439.40      |
| 10          | 439.50      |

The correlation coefficient[^1] for the data in table 1 is 0.999, very close to 1 indeed. So the points almost form a straight line (figure 5), just as Hooke's law predicts. The minor deviations from linearity are probably due to measurement error; neither the weights nor the length have been measured with perfect accuracy. (Nothing ever is. [When it comes to measurement])

![Hooke's law](./images/stats_hookes_law.png)

Our goal is to estimate $\hat{m}$ and $\hat{b}$ in the equation of Hooke's law for this spring:
$$
y = \hat{m} x + \hat{b}
$$
The graph of this equation is a perfect straight line. If the points in figure 5 happened to fall exactly on some line, the slope[^2] of that line would estimate $m$, and its intercept would estimate $b$. However, the points do not line up perfectly. Many different lines could be drawn across the scatter diagram, each having a slightly different slope and intercept.

Which line should be used? Hooke's equation predicts the length from weight. As discussed above, it is natural to choose $m$ and $b$ so as to minimize the r.m.s error, the line $y = \hat{m} x + \hat{b}$ which does the job is the **regression line**. This is the *method of least squares*. In other words, $m$ in Hooke's law should be estimated as the slope of the regression line, and $b$ as its intercept. These are called *least squares estimate*, because they minimize root-mean-square error.

Let's do the arithmetic (in python code):

```python
import numpy as np

# X the weight data; y the length data
X = np.array([0, 2, 4, 6, 8, 10])
y = np.array([439.00, 439.12, 439.21, 439.31, 439.40, 439.50])

# mean and Standard Deviation
# ---------------------------
# avg = sum(X) / len(X)
mu_x = X.mean()
mu_y = y.mean()
print(f"The means of X and y: {mu_x, mu_y}")
# The means of X and y: (5.0, 439.25666666666666)

# std is the "r.m.s size of the deviation from the average"
SD_x = X.std()
SD_y = y.std()
print(f"The SDs of X and y: {SD_x, SD_y}")
# The SDs of X and y: (3.415650255319866, 0.16799470891138593)

# convert X into standard unit form
X_standard_unit = (X - mu_x) / SD_x
# convert y into standard unit form
y_standard_unit = (y - mu_y) / SD_y
# correlation coefficient is the average of the products
r = (X_standard_unit.dot(y_standard_unit)) / len(X_standard_unit)
# r = 0.999167257319307

# the slope
m_hat = (r * SD_y) / SD_x
# m_hat = 0.0491428571428563

# the intercept, this is the *predicted length* when weight is 0,
b_hat = mu_y - (mu_x * m_hat)
# b_hat = 439.0109523809524
```

this gives us: $\hat{m} \approx 0.05$ per kg and $\hat{b} \approx 439.01$ cm.

The length of the spring under no load is estimated as 439.01 cm. And each kilogram of load causes the spring to stretch by about 0.05 cm. Of course, even Hooke's law has its limits: beyond some point, the spring will break. **Extrapolating beyond the data is risky**.

The method of least squares and the regression method involve the same mathematics; but the contexts may be different. In some fields, investigators talk about "least squares" when they are estimating parameters -- unknown constants of nature like $m$ and $b$ in Hooke's law. In other fields, investigators talk about regression when they are studying the relationship between two variables, like income and education, using non-experimental data.

**A technical point:** The least squares estimate for the length of the spring under no load was 439.01 cm. This is a tiny bit longer than the measured length at no load (439.00 cm). A statistician might trust the least squares estimate over the measurement. Why? Because the least squares estimate takes advantage of all six measurements, not just once: some of the measurement error is likely to cancel out. Of course, the six measurements are tied together by a good theory -- Hooke’s law. Without the theory, the least squares estimate wouldn’t be worth much.

[^1]: Convert each variable to standard units. The average of the products gives the correlation coefficient (may be more intuitively in the python code)
[^2]: Associated with a unit increase in $x$ there is some average change in $y$. The slope of the regression line estimates this change. The formula for the slope is $\frac{r \times SD_y}{SD_x}$. And the intercept of the regression line is just the predicted value for $y$ when $x$ is $0$.



### Assessing the Accuaracy of the Coefficient Estimates

Assume that the *true* relationship (e.g., the Hooke’s law) between $X$ and $Y$ takes the form $Y = f(X) + \epsilon$ for some unknown function $f$, where $\epsilon$ is a mean-zero random error term.  If $f$ is to be approximated by a linear function, then we can write this relationship as
$$
Y = \beta_0 + \beta_1 X + \epsilon.
$$
This is the *population regression line*. Here $\beta_0$ is the intercept (the expected value of $Y$ when $X$ = 0) and the $\beta_1$ the slop (the average increase in $Y$ associated with a one-unit increase in $X$). The $\epsilon$ (error term, typically assumed to be independent of $X$) is a catch-all for what we miss with this simple model: the true relationship is probably not linear, there may be other variables that cause variation in $Y$, and there may be measurement error.

The model of *population regression line* is the best linear approximation to the true relationship between $X$ and $Y$ (NOTE that the assumption of linearity is often a useful working model. However, it may be not true in reality). The true relationship is generally not known for real data, but the least squares line can always be computed using the cofficient estimation methods. A natural question is as follows: how accurate is the least square line as an estimate of the population regression line?

The analogy between linear regression and estimation of the mean of a random variable is an apt one based on the concept of *bias*. If we use the sample mean $\hat{\mu}$ to estimate $\mu$, this estimate is *unbiased*, in the sense that on average, we expect $\hat{\mu}$ to equal $\mu$, if we could average a huge number of estimates of $\mu$ obtained from a huge number of sets of observations. Hence, an unbiased estimator does not *systematically* over- or under-estimate the true parameter. The property of unbiasedness holds for the least squares coefficient estimates as well: if we estimate $\beta_0$ and $\beta_1$ on the basis of a particular data set, then our estimates won't be exactly equal to $\beta_0$ and $\beta_1$. But if we could average the estimates obtained over a huge number of date sets, then the average would be spot on!

So how far off will that single estimate of $\hat{\mu}$ be? In general, we answer this question by computing the *standard error* of $\hat{\mu}$, written as $SE(\hat{\mu})$. We have the well-known formula
$$
Var(\hat{\mu}) = SE(\hat{\mu})^2 = {\sigma^2 \over n},
$$
where $\sigma$ is the standard deviation of each of the realizations $y_i$ of $Y$. NOTE that this formula holds iff the $n$ observations are uncorrelated. To compute the standard errors associated with $\hat{\beta_0}$ and $\hat{\beta_1}$, we use the following formulas:
$$
SE(\hat{\beta_0})^2 =
\sigma^2 [{1 \over n} + \frac{\bar{x}^2}{\sum^n_{i=1}(x_i - \bar{x})^2}],
\
SE(\hat{\beta_1})^2 =\frac{\sigma^2}{\sum^n_{i=1}(x_i - \bar{x})^2}
$$
where $\sigma^2 = Var(\epsilon)$. In general, $\sigma^2$ is not known, but can be estimated from the data. This estimate of $\sigma$ is known as the *residual standard error*, and is given by the formula
$$
\sigma = RSE = \sqrt{RSS / (n-2)}
$$


#### Confidence Interval

Standard errors can be used to compute the *confidence intervals*. A 95% confidence interval is defined as a range of values such that with 95% probability, the range will contain the true unknown value of the parameters. The range is defined in terms of lower and upper limits computed from the sample of data.

For linear regression, the 95% confidence interval for $\beta_1$ approximately takes the form
$$
\hat{\beta_1} \pm 2 \cdot SE(\hat{\beta_1}).
$$
That is, there is approximately a 95% chance the true value of $\beta_1$ would be in this range.

Similarly, for $\beta_0$, its 95% confidence interval takes the form
$$
\hat{\beta_0} \pm 2 \cdot SE(\hat{\beta_0}).
$$
NOTE that here we make an assumption that the errors are Gaussian. And the factor of $2$ in the formula will vary slightly depending on the number of observations $n$ in the linear regression.

#### Hypothesis tests

Standard errors can also be used to perform *hypothesis tests* on the coefficients. The most common hypothesis test involves testing the *null hypothesis* of
$$
H_0 : \text{There is no relationship between X and Y}
$$
versus the *alternative hypothesis*
$$
H_a : \text{There is some relationship between X and Y}.
$$
Mathematically, this corresponds to testing
$$
H_0 : \beta_1 = 0
$$
versus
$$
H_a : \beta_1 \ne 0,
$$
since if $\beta_1 = 0$ then the linear regression model reduces to $Y = \beta_0 + \epsilon$, and $X$ is not associated with $Y$.

To test the null hypothesis, we need to determine whether $\hat{\beta_1}$, our estimate for $\beta_1$, is sufficiently far from zero that we can be confident that $\beta_1$ is non-zero. How far is far enough? This is of course depends on the accuracy of $\hat{\beta_1}$ -- that is, it depends on $SE(\hat{\beta_1})$:

- If $SE(\hat{\beta_1})$ is small, then even relatively small values of $\hat{\beta_1}$ may provide strong evidence that $\beta_1 \ne 0$;
- if $SE(\hat{\beta_1})$ is large, then $\hat{\beta_1}$ must be large in absolute value in order for us to reject the null hypothesis.

In practice, we compute a *t-statistic*, given by
$$
t = \frac{\hat{\beta_1} - 0}{SE(\hat{\beta_1})},
$$
which measures the number of standard deviations that $\hat{\beta_1}$ is away from $0$. 

If there really is no relationship between $X$ and $Y$, then we expect that *t-statistic* will have a $t$-distribution with $n-2$ degrees of freedom. Consequently, it is a simple matter to compute the probability of observing any number equal to $|t|$ or larger in absolute value, assuming $\beta_1 = 0$. We call this probability the *p-value*.

**p-value interpretation**

Roughly speaking, we interpret the p-value as follows: a small p-value indicates that it is unlikely to observe such a substantial association between the predictor and the response due to chance, in the absence of any real relationship between $X$ and $Y$. Hence we *reject the null hypothesis*, and declare a relationship to exist between $X$ and $Y$, if the p-value is small enough. Typical p-value cutoffs for rejecting the null hypothesis are 5% or 1%, when $n = 30$, these correspond to *t-statistics* of around $2$, and $2.75$, respectively.

### Assessing the Accuracy of the Model

Once we have rejected the null hypothesis in favor of the altervative hypothesis, it is natrual to want to quantify *the extent to which the model fits the data*. The quality of a linear regression fit is typically assessed using two related quantities: the *residual standard error* ($RSE$) and the $R^2$ statistic.

#### Residual Standard Error

From the model $Y = \beta_0 + \beta_1 X + \epsilon$ that associated with each observation is an error term $\epsilon$. Due to the presence of these error terms, even if we knew the true regression line (i.e., $\beta$s were known), we would not be perfectly predict $Y$ from $X$. The $RSE$ is an estimate of the standard deviation of $\epsilon$. Roughly speaking, it is the average amount that the response will deviate from the true regression line. It is computed using the formula
$$
RSE
= \sqrt{{1 \over n-2} RSS}
= \sqrt{{1 \over n-2} \sum^n_{i=1}(y_i - \hat{y_i})^2}.
$$
NOTE that $RSE$ is slightly different from *r.m.s error* which the latter using the number of all samples ($n$) as denominator while the former using $n-2$.

The $RSE$ is considered a measure of the *lack of fit* of the model to the data. The smaller $RSE$ the better the model fitted to the data.

#### $R^2$ Statistic

The $RSE$ provides an absolute measure of lack of fit of the model to the data. But since it is measured in the units of $Y$, it is not always clear what consititues a good $RSE$. The $R^2$ statistic provides an alternative measure of fit. It takes the form of a *proportion*, the proportion of variance explained, and so it always takes on a value between $0$ and $1$, and is independent of the scale of $Y$.

To calculate $R^2$, we use the formula
$$
R^2 = \frac{TSS - RSS}{TSS} = 1 - \frac{RSS}{TSS}
$$
where $TSS = \sum(y_i - \bar{y})^2$ is the *total sum of squares*. Hence $R^2$ measures the *proportion of variability in $Y$ that can be explained using $X$*.



### Multiple Linear Regression

Simple linear regression[^3] is a useful approach for predicting a response on the basis of a single predictor variable. But in practice we often have more than one predictor. One option is to run multiple separate simple linear regression, each of which uses a different feature as a predictor. However, this approach is not entirely satisfactory[^4]. 

Instead of fitting a separate simple linear regression model for each predictor, a better approach is to extend the simple linear regression model[^5] so that it can directly accommodate multiple predictors. We can do this by giving each predictor a separate slope coefficient in a single model. In general, suppose we have $p$ distinct predictors. Then the multiple linear regression model takes the form
$$
\tag{3.19}
\label{mlr}
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots +  + \beta_p X_p + \epsilon
$$
where $X_j$ represents the $j$-th predictor and $\beta_j$ quantifies the association between that variable and the response. We interpret $\beta_j$ as the *average* effect on $Y$ of a unit increase in $X_j$, **holding all other predictors fixed**[^6].

As was the case in the simple linear regression setting, the regression coefficients $\beta_0, \beta_1, \ldots, \beta_p$ in ($\ref{mlr}$) are unknown, and must be estimated. Given estimates $\hat{\beta_0}, \hat{\beta_1}, \ldots, \hat{\beta_p}$, we can make predictions using the formula
$$
\tag{3.21}
\label{mlrpred}
\hat{y} = \hat{\beta_0} + \hat{\beta_1} x_1 + \hat{\beta_2} x_2 + \cdots  + \hat{\beta_p} x_p.
$$
Then parameters are estimated using the same **least squares** approach that we saw in the context of simple linear regression. We choose $\beta_0, \beta_1, \ldots, \beta_p$ to minimize the **sum of squared residuals**
$$
\begin{eqnarray}
RSS 
&=& \sum^n_{i=1}(y_i - \hat{y}_i)^2 \\
\tag{3.22} \label{rss}
&=& \sum^n_{i=1} \big( y_i - (\hat{\beta_0} + \hat{\beta_1} x_1 + \hat{\beta_2} x_2 + \cdots  + \hat{\beta_p} x_p) \big)^2
\end{eqnarray}
$$
The values $\hat{\beta_0}, \hat{\beta_1}, \ldots, \hat{\beta_p}$ that minimize ($\ref{rss}$) are the multiple least squares regression coefficient estimates. Unlike the simple linear regression coefficient estimation (the Python code block in previous section), the multiple regression coefficient estimates have somewhat complicated forms that are most easily represented using matrix algebra (see detail in section of Normal equation).

When we perform multiple linear regression, we usually are interested in answering a few important questions.

---

1. Is at least one of the predictors $X_1, X_2, \ldots, X_p$ useful in predicting the response?
2. Do all the predictors help to explain $Y$, or is only a subset of the predictors useful?
3. How well does the model fit the data?
4. Given a set of predictor values, what response value should we predict, and how accurate is our prediction?

---

We now address each of these questions in turn.

#### One: Is There a Relationship Between the Response and Predictions?

Recall that in the simple linear regression setting, in order to determine whether there is a relationship between the response and the predictor we can simply check whether the slope equals $0$. In the multiple regression setting with $p$ predictors, we need to ask whether all of the regression coefficients are zero. We also use a **hypothesis test** to answer this question. We test the **null hypothesis**,
$$
H_0:\beta_0 = \beta_1 = \ldots = \beta_p = 0
$$
versus the alternative
$$
H_{\alpha}: \text{at least one } \beta_j \text{ is non-zero}
$$
This hypothesis test is performed by computing the **F-statistic**,
$$
\tag{3.23}
F = \frac{(TSS - RSS) / p}{RSS / (n - p -1)}
$$
where, as with simple linear regression, $TSS = \sum(y_i - \bar{y})^2$ and $RSS = \sum(y_i - \hat{y})^2$ where $\bar{y} = {1 \over n} \sum^n_{i=1} y_i$ is the sample mean. If the linear model assumptions are correct, one can show that
$$
E \{RSS / (n - p - 1) \} = \sigma^2
$$
and that, provided $H_0$ is true,
$$
E \{(TSS - RSS) / p \} = \sigma^2
$$
Hence, when there is no relationship between the response and predictors, one would expect the F-statistic to take on a value close to $1$. On the other hand, if $H_{\alpha}$ is true, then $E \{(TSS - RSS) / p \} > \sigma^2$, so we expect the F-statistic to be greater than $1$.

However, what if the F-statistic had been closer to $1$? How large does the F-statistic need to be before we can reject $H_0$ and conclude that there is a relationship? It turns out that the answer depends on the values of $n$ and $p$.

- When $n$ is large, an F-statistic that is just a little larger than 1 might still provide evidence against $H_0$.
- Inctrast, a larger F-statistic is needed to reject $H_0$ if $n$ is small.

When $H_0$ is true, and the errors $\epsilon_i$ have normal distribution, the F-statistic follows an F-distribution[^7]. For any given value of $n$ and $p$, any statistical software package can be used to compute the p-value[^8] associated with F-statistic using this distribution. Based on this p-value, we determine whether or not to reject $H_0$.

Sometimes we want to test that a particular subset of $q$ of the coefficients are zero. This corresponds to a null hypothesis
$$
H_0 = \beta_{p-q+1} = \beta_{p-q+2} = \cdots = \beta_{p} = 0,
$$
where for convenience we have put the variables chosen for omission at the end of the list. In this case we fit a second model that uses all the variables except those last $q$. Suppose that the residual sum of squares for that model is $RSS_0$. Then the appropriate F-statistic is
$$
\tag{3.24} \label{ftest2}
F = \frac{(RSS_0 - RSS) / q}{RSS / (n - p -1)}.
$$
For each individual predictor a t-statistic and a p-value can be obtain, these statistics provide information about whether each individual predictor is related to the response, after adjusting for the other predictors. It turns out that each of these are exactly equivalent to the F-test that omits that single variable from the model, leaving all the others in (means $q=1$ in equation $\ref{ftest2}$).  So it reports the *partial effect* of adding that variable to the model.

Given these individual p-values for each variable, why do we need to look at the over F-statistic? After all, it seems likely that if any one of the p-values for the individual variables is very small, then *at least one of the predictors is related to the response*. However, this logic is flawed, especially when the number of predictors $p$ is large.

For instance, consider an example in which $p = 100$ and $H_0:\beta_0 = \beta_1 = \ldots = \beta_p = 0$ is true, so no variable is truly associated with the response. In this situation, about **5%** of the p-values associated with each variable will be below 0.05 by chance. **In other words, we expect to see approximately five small p-values even in the absence of any true association between the predictors and the response. In fact, we are almost guaranteed that we will observe at least one p-value below 0.05 by chance!**

Hence, if we use individual t-statistic and the associated p-value in order to confirm the association between any predictor and the response, there is a very high chance that we will incorrectly conclude that there is a relationship.

However, the F-statistic does not suffer from this problem because it adjusts for the number of predictors. If $H_0$ is true, there is only a 5% chance that the F-statistic will result in a p-value below 0.05, regardless of the number of predictors or the number of observations.

Note that when $p$ is larger than $n$, we cannot even fit the multiple linear regression model using least squares. Less flexible least squares models, such as forward stepwise selection, ridge regression, lasso regression and principal components regression, are particular useful for performing regression in the high-dimensional setting.

#### Two: Deciding on Important Variable

If we conclude on the basis of the F-statistic and its associated p-value that at least one of the predictors is related to the response, then it is natural to wonder which are the guilty ones. The task of determining which predictors are associated with the response, in order to fit a single model involving only those predictors, is referred to as **variable selection**. 

Ideally, we would like to preform variable selection by trying out a lot of different models, each containing a different subset of the predictors. Unfortunately, there are a total $2^p$ models that contain subsets of $p$ variables (Note that even with a moderate value of $p$, say, $p=30$, then $2^{30}=1,073,741,824$ models make this infeasible). We need an automated and efficient approach to choose a smaller set of models to consider. There are three classical approaches for this task:

- Forward selection. We begin with the **null model** (which contains only the intercept). We then fit $p$ simple linear regressions and add to the null model the variable that results in the lowest $RSS$, and then add to that model the variable which results in the lowest $RSS$ for the new two-variable model. This approach is continued until some stopping rule is satisfied.
- Backward selection. We start with all variables in the model, and remove the variable with largest p-value. The new ($p - 1$)-variable model is fit, and the variable with the largest p-value is removed. This procedure continues until a stopping rule is reached (such as when all remaining variables have a p-value below some threshold).
- Mixed selection. We start with no variables in the model, and as with forward selection, we add the variable that provides the best fit. We continue to add variables one-by-one. If at any point the p-value for one of the variables in the model rises above a certain threshold, then we remove that variable from the model. We continue to perform forward and backward steps until all variable in the model have a low p-value, and all the variables outside the model have a large p-value if added to the model.

Backward selection cannot be used when $p > n$, forward selection is a greedy approach, and might include variable early that later become redundant. Mixed selection can remedy this.

#### Three: Model Fit

Two of the most common numerical measures of model fit are the $RSE$ and the $R^2$. Recall that in simple regression, $R^2$ is the square of the correlation coefficient between predictor and the response. In multiple linear regression, it turns out that it equals the square of the correlation coefficient between the response and the fitted model (this implies that the fitted model maximizes this correlation among all possible linear model).

To calculate $R^2$, we use the formula
$$
\begin{eqnarray}

R^2
&=& \frac{\sum(y_i - \bar{y})^2 - \sum(y_i - \hat{y})^2}{\sum(y_i - \bar{y})^2} \\
\\
&=& \frac{TSS - RSS}{TSS} \\
\\
\tag{3.17}
&=& 1 - {RSS \over TSS},

\end{eqnarray}
$$
where ($\bar{y} = {1 \over n} \sum^n_{i=1} y_i$) is the sample mean, $\hat{y}$ is defined in ($\ref{mlrpred}$).

An $R^2$ value close to 1 indicates that the model explains a large portion of the variance in the response variable. It turns out that $R^2$ will always increase when more variables are added to the model, even they are only weakly associated with the response. This is due to the fact that with more variable to the least squares equations must allow us to fit the training data more accurately (though not necessarily the testing data, a.k.a., over fitting).

In general, $RSE$ is defined as
$$
\tag{3.25}
RSE = \sqrt{{1 \over {n - p - 1}} RSS}
$$
Thus model with more variables can have higher $RSE$ if the decrease in $RSS$ is small relative to the increase in $p$.

##### Adjusted $R^2$ [(From wiki)](https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2)

The use of an adjusted $R^2$ (one common notation is $\bar{R}^2$; another is $R_{adj}^2$) is an attempt to account for the phenomenon of the $R^2$ automatically and spuriously increasing when extra explanatory variables are added to the model. There are many different ways of adjusting, by far the most used one, to the point that it is typically just referred to as *adjusted $R^2$*, is the correction prosposed by Mordecai Ezekiel, and adjusted $R^2$ is defined as
$$
\bar{R}^2 = 1 - (1 - R^2){n-1 \over n-p-1}
$$
where $p$ is the total number of explantory variables in the model (not including the constant term), and $n$ is the sample size. It can also be written as
$$
\bar{R}^2 = 1 - \frac{RSS / df_e}{TSS / df_t}
$$
where $df_t$ is the *degrees of freedom* $n-1$ of the estimate of the population variance of the dependent variable, and the $df_e$ is the degrees of freedom $n-p-1$ of the estimate of the underlying population error variance.

The adjusted $R^2$ can be negative, and its value will always be less than or equal to that of $R^2$. Unlike $R^2$, the adjusted $R^2$ increases only when the increase in $R^2$ (due to the inclusion of a new variable) is more than one would expect to see by chance. If a set of explanatory variables with a predtermined hierarchy of importance are introduced into a regression one at a time, with the adjusted $R^2$ computed each time, the level at which adjusted $R^2$ reaches a maximum, and decreases afterward, would be the regression with ideal combination of having the best fit without excess/unnecessary terms.

> degrees of freedom [(From wiki)](https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics))
>
> The number of *degrees of freedom* is the number of values in the final calculation of a statistic that are free to vary.
>
> Estimates of statistical parameters can be based upon different amounts of information or data. The number of independent pieces of information that go into the estimate of a parameter are called the degrees of freedom.
>
> Mathematically, degrees of freedom is the number of dimensions of the domain of a random vector, or essentially the number of "free" components (how many components need to be know before the vector is fully determined).
>
> Suppose we have a sample of independent normally distributed observations, $\{X_1, X_2, \ldots, X_n\}$. This can be represented as an n-dimensional random vector:
>
> $X^T$. Since this random vector can lie anywhere in n-dimensional space, it has $n$ degrees of freedom.
>
> Now let $\bar{X}$ be the sample mean. The random vector can be decomposed as the sum of the sample mean plus a vector of residuals:
> $$
> \left( \begin{array}{c}
>     X_{1} \\
>     \vdots \\
>     X_{n} \\
> \end{array} \right)
> 
> = 
> \bar{X} \cdot
> \left( \begin{array}{c}
>     1 \\
>     \vdots \\
>     1 \\
> \end{array} \right)
> 
> +
> 
> \left( \begin{array}{c}
>     X_{1} - \bar{X} \\
>     \vdots \\
>     X_{n} - \bar{X}\\
> \end{array} \right).
> $$
> The first vector on the right-hand side is constrained to be a multiple of the vector of $1$'s, and the only free quantity is $\bar{X}$. It therefore has only one degree of freedom.
>
> The second vector is constrained by the relation $\sum(X_i - \bar{X}) = 0$. The first $n-1$ components of this vector can be anything. However, once you know the first $n-1$ components, the constraint tells you the value of the $n$th component. Therefore, this vector has $n-1$ degrees of freedom.

#### Four: Prediction

Once we have fit the multiple regression model, it is straightforward to apply the fitted model $\hat{y} = \hat{f}(X) = \hat{\beta} X$ (a more verbose version see $\ref{mlrpred}$) in order to predict the response based on the values of the predictors. However, there are three sorts of uncertainty associated with this prediction.

1. The coefficient estimate is the least squares estimation of the true coefficient which is unknown. The inaccuracy in the coefficient estimates is related to the *reducible error*[^9]. We can compute a **confidence interval** in order to determine how close $\hat{y}$ will be to $f(X)$.
2. In practice assuming a linear model for $f(X)$ is almost always an approximation of reality, so if the true pattern is non-linear, there is an additional reducible error called *model bias*.
3. Even if we knew $f(X)$ -- that is, we knew the true value of $\beta$ -- the response value cannot be predicted perfectly, because of the random error $\epsilon$ in the model ($\ref{mlrpred}$), this is the *irreducible error*.



### Summary

- The regression line can be specified by two descriptive statistics: the *slope* and the *intercept*.
- Among all lines, the regression line for $y$ on $x$ makes the smallest r.m.s error in predicting $y$ from $x$. For that reason, the regression line is often called the *least squares line*.

- With a controlled experiment, the slope can tell you the average change in $y$ that would be caused by a change in $x$. With an observational study, however, the slope cannot be relied on to predict the results of interventions. It takes a lot of hard work to draw causal inferences from observational data, with or without regression.
- If the average of $y$ depends on $x$ in a non-linear way, the regression line can be quite misleading.
- Multiple regression is a powerful technique, but it is not a substitute for understanding. (Such as the poor investigator would fit a multiple regression equation of the form $ predicted\ area = a + b \times perimeter + c \times diagonal$ to predict the area of a rectangle).




## Part-I: How to learn

Linear regression is of course an extremely simple and limited learning algorithm, but it provides an example of how a learning algorithm can work.

### Normal Equation

The goal is to build a system that can take a vector $x \in \mathbb{R^n}$ as input and predict the value of a scalar $y \in \mathbb{R}$ as its output.  The output of linear regression is a linear function of the input. Let $\hat{y}$ be the value that our model predicts $y$ should take on. We define the output to be
$$
\begin{equation}
\tag{5.3}
\hat{y} = w^{\mathsf{T}}x
\end{equation}
$$

where $w \in \mathbb{R^n}$ is a vector of **parameters**.

We thus have a definition of our task *T* : to predict $y$ from $x$ by outputting $\hat{y}=w^{\mathsf{T}}x$. 

Next we need a definition of our performance measure, *P*. One way of measuring the performance of the model is to compute the **mean squared error (MSE)** of the model on the test set. If $\hat{y}^{(test)}$ gives the predictions of the model on the test set, then the MSE is given by
$$
\tag{5.4}
MSE_{test} = \frac{1}{m} \sum_i{(\hat{y}^{(test)} - {y}^{(test)})^2_i}
$$
Intuitively, one can see that this error measure decreases to 0 when $\hat{y}^{(test)} = {y}^{(test)}$. We can also see that
$$
\tag{5.5}
MSE_{test} = \frac{1}{m} ||{\hat{y}^{(test)} - {y}^{(test)}}||^2_2
$$
so the error increases whenever the Euclidean distance between the predictions and the targets increases.

> In machine learning, we usually measure the size of vectors using a function called a **norm**. Formally, the $L^p$ norm is given by
> $$
> ||x||_p = \bigg(\sum_i |x_i|^p \bigg)^{\frac{1}{p}}
> $$
> for $p \in \R, p \geq 1$.
>
> The $L^2$ norm, with $p = 2$, is known as the **Euclidean norm**, often denoted simply as $||x||$. It is also common to measure the size of a vector using the squared $L^2$ norm, which can be calculated simply as $x^{\mathsf{T}}x$.

To make a machine learning algorithm, we need to design an algorithm that will improve the weights $w$ in a way that reduces $MSE_{test}$ when the algorithm is allowed to gain experience by observing a training set ($X^{(train)}, y^{(train)}$).

One intuitive way of doing this is  (to minimize $MSE_{test}$) just to minimize the MSE on the training set, $MSE_{train}$ .  (Does this make any sense? Keep on reading.)

To minimize $MSE_{train}$ , we can simply solve for where its gradient is **0**:
$$
\tag{5.6}
\nabla_w MSE_{train} = 0 \\
\Rightarrow \frac{1}{m} \nabla_w ||{\hat{y}^{(test)} - {y}^{(test)}}||^2_2 = 0 \\
\Rightarrow \frac{1}{m} \nabla_w ||{\hat{y}^{(train)} - {y}^{(train)}}||^2_2 = 0 \\
\Rightarrow \frac{1}{m} \nabla_w ||{X^{(train)}w - y^{(train)}}||^2_2 = 0 \\
\Rightarrow \nabla_w \big({X^{(train)}w - y^{(train)}}\big)^{\mathsf{T}} 				\big({X^{(train)}w - y^{(train)}}\big) = 0 \\
$$

$$
\tag{5.10}
\Rightarrow \nabla_w \big( w^{\mathsf{T}} X^{(train)\mathsf{T}} X^{(train)}w - 			2w^{\mathsf{T}} X^{(train)\mathsf{T}} y^{(train)} + y^{(train)\mathsf{T}} 			y^{(train)} \big) = 0 \\
$$

$$
\tag{5.11}
\Rightarrow 2X^{(train)\mathsf{T}} X^{(train)}w - 2X^{(train)\mathsf{T}} y^{(train)} 	= 0 \\
$$

$$
\tag{5.12}
\Rightarrow w = \big(X^{(train)\mathsf{T}} X^{(train)}\big)^{-1} X^{(train)\mathsf{T}} y^{(train)}
$$

The system of equations whose solution is given by equation 5.12 is known as the **normal equation**. Evaluating equation 5.12 constitutes a simple learning algorithm.

> 链式法则(Chain Rule)是计算复杂导数时的重要工具。简单地说，若函数 $f(x) = g(h(x))$，则有
> $$
> \tag{A.31}
> \frac{\partial{f(x)}}{\partial{x}} = \frac{\partial{g(h(x))}}{\partial{h(x)}} \cdot 	\frac{\partial{h(x)}}{\partial{x}}.
> $$

> 例如在计算下式时，将$(Ax - b)$看作一个整体可简化计算：
> $$
> \begin{align*}
> \frac{\partial}{\partial{x}}(Ax - b)^{\mathsf{T}} W(Ax - b) &= \frac{\partial{(Ax - b)}}{\partial{x}} \cdot 2W(Ax - b) \\
> &= 2AW(Ax - b)
> \end{align*}
> $$

It is worth noting that the term **linear regression** is often used to refer to a slightly more sophisticated model with one additional parameter -- an intercept term $b$. In this model
$$
\tag{5.13}
\hat{y} = w^{\mathsf{T}}x + b
$$
so the mapping from parameters to predictions is still a linear function but the mapping from features to predictions is now an **affine function** (which is in the form of equation 5.13, it means that the plot of model's predictions still looks like a line, but it need not pass though the origin). One can continue to use the model with only weights but augment $x$ with an extra entry that is always set to 1. Then the weight corresponding to the extra 1 entry plays the role of the bias parameter (i.e., the intercept term, $b$, a.k.a., the bias term).

> The intercept term $b$ is often called the **bias** parameter of the affine transformation. This terminology derives from the point of view that the output of the transformation is biased toward being $b$ in the absence of any input.
>
> This term is different from the idea of a statistical bias, in which a statistical estimation algorithm's expected estimate of a quantity is not equal to the true quantity.



### Gradient-based method

Example: Linear Least Squares

Suppose we want to find the value of $x$ that minimizes
$$
\tag{4.21}
f(x) = \frac{1}{2}||Ax - b||^2_2
$$
Specialized linear algebra algorithms can solve this problem efficiently; however, we can also explore how to solve it using gradient-based optimization as a simple example of how these techniques work.

First, we need to obtain the gradient (Recall the Chain Rule):
$$
\tag{4.42}
\nabla_x f(x) = A^{\mathsf{T}} (Ax - b) = A^{\mathsf{T}}Ax - A^{\mathsf{T}}b
$$
We can then follow this gradient downhill, taking small steps.

***

Algorithm 4.1 An algorithm to minimize $f(x) = \frac{1}{2}||Ax - b||^2_2$ with respect to $x$ using gradient descent, starting from an arbitrary value of $x$.

***

​	Set the step size ($\epsilon$, a.k.a., learning rate) and tolerance ($\delta$) to small, positive numbers.

​	**while** $||A^{\mathsf{T}}Ax - A^{\mathsf{T}}b||_2 > \delta$ **do**

​		$x \leftarrow x - \epsilon (A^{\mathsf{T}}Ax - A^{\mathsf{T}}b)$

​	**end while**

***



> 关于梯度下降法(Gradient Descent)
>
> 梯度下降法是一种常用的一阶(first-order)优化方法, 是求解无约束优化问题最简单,最经典的方法之一.
>
> 考虑无约束优化问题$min_x f(x)$,其中$f(x)$为连续可微函数.若能构造一个序列$x^0, x^1, x^2, \ldots$ 满足
> $$
> \tag{B.15}
> \label{eq_ngd}
> f(x^{(t+1)}) < f(x^{(t)}), t = 0,1,2,\ldots
> $$
> 则不断执行该过程即可收敛到局部极小点.欲满足式($\ref{eq_ngd}$),根据泰勒展式有
> $$
> \tag{B.16}
> f(x + \Delta x) \simeq f(x) + \Delta x^{\mathsf{T}} \nabla f(x)
> $$
> 于是,欲满足$f(x + \Delta x) < f(x)$,可选择
> $$
> \tag{B.17}
> \Delta x = - \epsilon \nabla f(x)
> $$
> 其中步长(step size)$\epsilon$是一个小常数.这就是梯度下降法.
>
> 若目标函数$f(x)$满足一些条件,则通过选取合适的步长,就能确保通过梯度下降收敛到局部极小点.例如,若$f(x)$满足L-Lipschitz条件(亦即,对于任意$x$,存在常数$L$使得$||\nabla f(x)|| \leq L$成立),则将步长设置为$1/(2L)$即可确保收敛到局部极小点.当目标函数是凸函数时,局部极小点就是全局最小点,此时,梯度下降法可确保收敛到全局最优解.
>
> 当目标函数$f(x)$二阶连续可微时,可将式($B.16$)替换成更为精确的二阶泰勒展式,这样就得到了牛顿法(Newton's method).牛顿法是典型的二阶方法,其迭代轮数远小于梯度下降法.但牛顿法使用了二阶导数$\nabla^2 f(x)$ (second derivative),其每轮迭代中涉及到海森矩阵(Hessian matrix)的求逆,计算复杂度相当高,尤其在高维问题中几乎不可行.其次,牛顿法仅适用于附近点有局部极小点的情况(也就是,海森矩阵为正定矩阵,也就是海森矩阵所有的特征值都是正数),若附近点是鞍点(saddle point)则牛顿法失效.然而,梯度下降却不会被鞍点困住.若能以较低的计算代价寻找海森矩阵的近似逆矩阵,则可以显著降低计算开销,这就是拟牛顿法(quai-Newton method).

### Newton's method

Sometimes we need to find all the partial derivatives of a function whose input and output are both vectors. The matrix containing all such partial derivatives is known as a **Jacobian matrix**. Specifically, if we have a function $f: \mathbb{R}^m \rightarrow \mathbb{R}^n$, then the Jacobian matrix $\mathbf{J} \in \mathbb{R}^{m \times n}$ of $f$ is defined such that $J_{i, j} = \frac{\partial}{\partial x_j}f(x)_i$.

We are also sometimes interested in a derivative of a derivative. This is known as a **second derivative**. For example, for a function $f : \mathbb{R}^n \rightarrow \mathbb{R}$, the derivative with respect to $x_i$ of the derivative of $f$ with respect to $x_j$ is denoted as $\frac{\partial^2}{\partial x_i \partial x_j}f$.

In a single dimension, we can denote $\frac{d^2}{d x^2}$ by $f''(x)$. The second derivative tells us how the first derivative will change when we vary the input. This is important because it tells us whether a gradient step will cause as much of an improvement as we would expect based on the gradient alone. We can think of the second derivative as measuring **curvature**.

Suppose we have a quadratic function (or in practice it can be approximated well as quadratic, at least locally). If such a function has:

- Second derivative of zero:  there is no curvature, it is a flat line, its value can be predict using only the gradient. If the gradient is 1,  set the step size of $\epsilon$ along the negative gradient, then the cost function will decrease by $\epsilon$.
- Second derivative is negative: the function curves downward, decrease by more than $\epsilon$.
- Second derivative is positive: the function curves upward, decrease by less than $\epsilon$.

When our function has multiple input dimensions, there are many second derivatives. These derivatives can be collected together into a matrix called the **Hessian matrix**. The Hessian matrix $H(f)(x)$ is defined such that
$$
\tag{4.6}
H(f)(x)_{i, j} = \frac{\partial^2}{\partial x_i \partial x_j}f(x).
$$
Equivalently, the Hessian is the Jacobian of the gradient.

The (directional) second derivative tells us how well we can expect a gradient descent step to perform. We can make a second-order Taylor series approximation to the function $f(x)$ around the current point $x^{(0)}$:
$$
\tag{4.8}
f(x) \approx f(x^{(0)}) + (x - x^{(0)})^{\mathsf{T}}g + \frac{1}{2} (x - x^{(0)})^{\mathsf{T}}H(x - x^{(0)}),
$$
where $g$ is the gradient and $H$ is the Hessian at $x^{(0)}$. If we use a learning rate of $\epsilon$, then the new point $x$ will be given by $x^{(0)} - \epsilon g$. Substituting this into our approximation, we obtain
$$
\tag{4.9}
f(x^{(0)} - \epsilon g) \approx f(x^{(0)}) - \epsilon g^{\mathsf{T}}g + \frac{1}{2} \epsilon^2 g^{\mathsf{T}}Hg.
$$
There are three terms here:

- the original value of the function
- the expected improvement due to the slope of the function
- and the correction we must apply to account for the curvature of the function

When this last term is too large, the gradient descent step can actually move uphill.

When $g^{\mathsf{T}}Hg$ is zero or negative, the Taylor series approximation predicts that increase $\epsilon$ forever will decrease $f$ forever. In practice, the Taylor series is unlikely to remain accurate for large $\epsilon$, so one must resort to more heuristic choices of $\epsilon$ in this case.

When $g^{\mathsf{T}}Hg$ is positive, solving for the optimal step size that decrease the Taylor series approximation of the function the most yields
$$
\tag{4.10}
\epsilon^* = \frac{g^{\mathsf{T}}g}{g^{\mathsf{T}}Hg}.
$$
In the worst case, when $g$ aligns with the eigenvector of $H$ corresponding to the maximal eigenvalue $\lambda_{max}$, then this optimal step size is given by $\frac{1}{\lambda_{max}}$. (The eigenvalues of the Hessian determine the scale of the learning rate, if the function we minimized can be approximated well by a quadratic function.)

Using the eigendecomposition of the Hessian matrix, we can generalize the **second derivative test** to multiple dimensions. At a critical point, where $\nabla_x f(x) = 0$, we can examine the eigenvalues of the Hessian to determine whether the critical point is a local maximum, local minimum, or saddle point.

- when the Hessian is positive definite (all its eigenvalues are positive): local minimum.
- when the Hessian is negative definite (all its eigenvalues are negative): local maximum.
- the test is inconclusive whenever all the nonzero eigenvalues have the same sign but at least one eigenvalue is zero.

In multiple dimensions, there is a different second derivative for each direction at a single point. The **condition number** of the Hessian at this point measures how much the second derivative differ from each other. When the Hessian has a poor condition number, gradient descent performs poorly. This is because in one direction, the derivative increases rapidly, while in another direction, it increases slowly, Gradient descent is unaware of this change in the derivative, so it does not know that it needs to explore preferentially in the direction where the derivative remains negative for longer.

Poor condition number also makes choosing a good step size difficult. The step size must be small enough to avoid overshooting the minimum and going uphill in directions with strong positive curvature. This usually means that the step size is too small to make significant progress in other directions with less curvature.

This issue can be resolved by using information from the Hessian matrix to guide the search. The simplest method for doing so is known as **Newton's method**. Newton's method is based on using a second-order Taylor series expansion to approximate $f(x)$ near some point $x^{(0)}$:
$$
\tag{4.11}
f(x) \approx f(x^{(0)}) + (x - x^{(0)})^{\mathsf{T}} \nabla_x f(x^{(0)}) + \frac{1}{2}(x - x^{(0)})^{\mathsf{T}}H(f)(x^{(0)})(x - x^{(0)})
$$
If we solve for the critical point of this function, we obtain
$$
\tag{4.12}
x^* = x^{(0)} - H(f)(x^{(0)})^{-1} \nabla_x f(x^{(0)})
$$
When $f$ is a positive definite quadratic function, Newton's method consist of applying equation 4.12 once to jump to the minimum of the function directly. When $f$ is not truly quadratic but can be locally approximated as a positive definite quadratic, Newton's method consists of applying equation 4.12 multiple times. NOTE that Newton's method is only appropriate when the nearby critical point is a minimum (all the eigenvalues of the Hessian are positive), whereas gradient is not attracted to saddle points unless the gradient points toward them.

## Part_II: Why that  Work

### Linear Regression as Maximum Likelihood

Previously, we motivated linear regression as an algorithm that learns to take an input $x$ and produce an output value $\hat{y}$. The mapping from $x$ to $\hat{y}$ is chosen to minimize mean squared error, a criterion that we introduced more or less arbitrarily. We now revisit linear regression from the point of view of maximum likelihood estimation.

Instead of producing a single prediction $\hat{y}$, we now think of the model as producing a conditional distribution $p(y | x)$. We can imagine that with an infinitely large training set, we might see several training examples with the same input value $x$ but different values of $y$. The goal of the learning algorithm is now to fit the distribution $p(y | x)$ to all those different $y$ values that are all compatible with $x$.

To derive the same linear regression algorithm we obtained before, we **define** $p(y | x) = \mathcal{N}(y; \hat{y}(x; w), \sigma^2)$. In this example, we assume that the variance is fixed to some constant $\sigma^2$ chosen by user.

Since the examples are assumed to be i.i.d., the conditional log-likelihood is given by 
$$
\begin{split}
&\sum^m_{i=1} log \ p(y^{(i)}|x^{(i)}; \theta) \\
&= -m \ log \ \sigma - \frac{m}{2} log(2 \pi) - \sum^m_{i=1} \frac{||\hat{y}^{(i)} - {y}^{(i)}||^2}{2 \sigma^2},
\end{split}
$$
where $\hat{y}^{(i)}$ is the output of the linear regression on the $i$-th input $x^{(i)}$ and m is the number of the training examples. Comparing the log-likelihood with the mean squared error,
$$
MSE_{train} = \frac{1}{m} \sum^m_{i=1} ||\hat{y}^{(i)} - {y}^{(i)}||^2,
$$
we immediately see that maximizing the log-likelihood with respect to $w$ yields the same estimate of the parameters $w$ as does minimizing the mean squared error. The two criteria have different values but the same location of the optimum.

This justifies the use of MSE as a maximum likelihood estimation procedure.

#### Maximum Likelihood Estimation

> Rather than guessing that some function might make a good estimator and then analyzing its bias and variance, we would like to have some principle from which we can derive specific functions that are good estimators for different models.
>
> The most common such principle is the maximum likelihood principle.
>
> Consider a set of m examples $\mathbb{X} = \{x^{(1)}, \cdots, x^{(m)}\}$ are i.i.d from true but unknown data-generating distribution $p_{data}(\mathbf{x})$.
>
> Let $p_{model}(\mathbf{x}; \mathbf{\theta})$ be a parametric family of probability distribution over the same space indexed by $\mathbf{\theta}$. In other words, $p_{model}({x}; \mathbf{\theta})$ maps any configuration $x$ to a real number estimating the true probability $p_{data}({x})$.
>
> The maximum likelihood estimator for $\mathbf{\theta}$ is then defined as
> $$
> \begin{eqnarray}
> \mathbf{\theta_{ML}} 
> \tag{5.56}
> &=& \underset{\mathbf{\theta}}{\operatorname{argmax}} {p_{model}(\mathbb{x}; \mathbf{\theta})} \\
> \tag{5.57}
> &=& \underset{\mathbf{\theta}}{\operatorname{argmax}} \prod^m_{i=1}{p_{model}(x^{(i)}; \mathbf{\theta})}
> \end{eqnarray}
> $$
> This product over many probabilities can be inconvenient for various reasons. Such as it's prone to numerical underflow. We observe that taking the logarithm of the likelihood does not change its argmax but does conveniently transform a product into a sum:
> $$
> \tag{5.58}
> \mathbf{\theta_{ML}} = \underset{\mathbf{\theta}}{\operatorname{argmax}} \sum^m_{i=1}{\text{log} \ p_{model}(x^{(i)}; \mathbf{\theta})}
> $$
> Because the argmax does not change when we rescale the cost function, we can divide by $m$ to obtain a version of the criterion that is expressed as an expectation with respect to the empirical distribution $\hat{p}_{data}$ defined by the training data:
> $$
> \tag{5.59}
> \mathbf{\theta_{ML}} = \underset{\mathbf{\theta}}{\operatorname{argmax}} \mathbb{E}_{\mathbf{x} \sim \hat{p}_{data}}{\text{log} \ p_{model}(x^{(i)}; \mathbf{\theta})}
> $$
> One way to interpret maximum likelihood estimation is to view it as minimizing the dissimilarity between the empirical distribution $\hat{p}_{data}$, defined by the training set and the model distribution, with the degree of dissimilarity between the two measured by the KL divergence. The KL divergence is given by
> $$
> \tag{5.60}
> D_{KL}(\hat{p}_{data} || p_{model}) = \mathbb{E}_{\mathbf{x} \sim \hat{p}_{data}} {[\text{log} \ \hat{p}_{data}(x) - \text{log} \ {p}_{model}(x)]}.
> $$
> The term on the left is a function only of the data-generating process, not the model. This means when we train the model to minimize the KL divergence, we need only minimize
> $$
> \tag{5.61}
> -\mathbb{E}_{\mathbf{x} \sim \hat{p}_{data}} {[\text{log} \ \hat{p}_{data}(x)]},
> $$
> which is of course the same as the maximization in equation 5.59.
>
> Minimizing this KL divergence corresponds exactly to minimizing the cross-entropy between the distributions. Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution and the model distribution. For example, MSE is the cross-entropy between the empirical distribution and a Gaussian model.
>
> We can thus see maximum likelihood as an attempt to make the model distribution match the empirical distribution $\hat{p}_{data}$. While the optimal $\mathbf{\theta}$ is the same regardless of whatever we are maximizing the likelihood or minimizing the KL divergence, the values of the objective functions are different.
>
> In software, we often phrase both as minimizing a cost function.
>
> Maximum likelihood thus becomes minimization of the negative log-likelihood (NLL), or equivalently, minimization of the cross-entropy.
>
> 关于KL散度
>
> KL散度(Kullback-Leibler divergence), 亦称相对熵(relative entropy)或信息散度(information divergence), 可用于度量两个概率分布之间的差异. 给定两个连续型概率分布$P$和$Q$, 二者之间的KL散度定义为
> $$
> \begin{equation} \label{eq_kld}
> \tag{C.34}
> KL(P||Q) = \int^{\infin}_{-\infin}p(x)\text{log}\frac{p(x)}{q(x)}\text{d}x,
> \end{equation}
> $$
> 其中,$p(x)$和$q(x)$分别是$P$和$Q$的概率密度函数.
>
> KL散度满足非负性, 即
> $$
> \tag{C.35}
> KL(P||Q) \geq 0,
> $$
> 当且仅当$P=Q$时$KL(P||Q)=0$. 但是, KL散度不满足对称性, 即
> $$
> \tag{C.36}
> KL(P||Q) \neq KL(Q||P),
> $$
> 因此, KL散度不是一个度量(metric).
>
> 若将KL散度的定义($\ref{eq_kld}$)展开, 可得
> $$
> \begin{eqnarray}
> 
> KL(P||Q) 
> &=& \int^{\infin}_{-\infin} p(x)\text{log} \ p(x)\text{d}x - \int^{\infin}_{-\infin} p(x)\text{log} \ q(x)\text{d}x \\
> \tag{C.37}
> &=& -H(P) + H(P, Q),
> \end{eqnarray}
> $$
> 其中$H(P)$为熵(entropy), $H(P,Q)$为交叉熵(cross-entropy).



### Bayesian Linear Regression

So far we have discussed **frequentist statistics** and approaches based on estimating a single value of $\theta$, then making all predictions thereafter based on that one estimate. An other approach is to consider all possible values of $\theta$ when making a prediction. The latter is the domain of **Bayesian statistics**.

- Freqentist: the true parameter value $\theta$ is fixed but unknown, while the point estimate $\hat{\theta}$ is a random variable on account of it being a function of the dataset (which is seen as random).
- Bayesian: the Bayesian uses probability to reflect degrees of certainty in states of knowledge. The dataset is directly observed and so is not random. On the other hand, the true parameter $\theta$ is unknown or uncertain and thus is represented as random variable.

Before observing the data, we represent our knowledge of $\theta$ using the **prior probability distribution**, $p(\theta)$ (a.k.a., "the prior"). Generally, the machine learning practitioner selects a prior distribution that quite broad (i.e., with high entropy, such as uniform distribution) to reflect a high degree of uncertainty in the value of $\theta$ before observing any data.

Now consider that we have a set of data samples {$x^{(1)}, \ldots, x^{(m)}$}. We can recover the effect of data on our belief about $\theta$ by combining the data likelihood $p(x^{(1)}, \ldots, x^{(m)} | \theta)$ with the prior via Bayes' rule:
$$
\tag{5.67}
p(\theta | x^{(1)}, \ldots, x^{(m)}) = \frac{p(x^{(1)}, \ldots, x^{(m)} | \theta) p(\theta)}{p(x^{(1)}, \ldots, x^{(m)})}
$$

In the scenarios where Bayesian estimation is typically used, the prior begins as a relatively uniform or Gaussian distribution with high entropy, and the observation of the data usually causes the posterior to lose entropy and concentrate around a few highly likely values of the parameters.

Bayesian estimation offers two important differences from MLE:

1. Unlike the MLE approach that makes predictions using a point estimate of $\theta$, the Bayesian approach is to make predictions using a full distribution over $\theta$. For example, after observing $m$ examples, the predicted distribution over the next data sample, $x^{(m+1)}$, is given by
   $$
   \tag{5.68}
   p(x^{(m+1)} | x^{(1)}, \ldots, x^{(m)}) = \int p(x^{(m+1)} | \theta) p(\theta | x^{(1)}, \ldots, x^{(m)}) d \theta
   $$
   Here each value of $\theta$ with positive probability density contributes to the prediction of the next example, with the contribution weighted by the posterior density itself.

   After having observed {$x^{(1)}, \ldots, x^{(m)}$}, if we are still uncertain about the value of $\theta$, then this uncertainty  is incorporated into any predictions we might make.

2. The prior has an influence by shifting probability mass density towards regions of the parameter space that are preferred a priori. In practice, the prior often expresses a preference for models that are simpler or more smooth.

Critics of the Bayesian approach identify the prior as a source of subjective human judgment affecting the predictions.

Bayesian methods typically generalize much better when limited training data is available but typically suffer from high computational cost when the number of training examples is large.

**$\color{Green}{\mathbf{Example}}$**

Here we consider the Bayesian estimation approach to learning the linear regression parameters. In linear regression, we learn a linear mapping from an input vector $x \in \mathbb{R}^n$ to predict the value of  a scalar $y \in \mathbb{R}$. The prediction is parameterized by the vector $w \in \mathbb{R}^n$:
$$
\tag{5.69}
\hat{y} = w^{\mathsf{T}}x.
$$
Given a set of $m$ training samples ($X^{(train)}, y^{(train)}$), we can express the prediction of $y$ over the entire training set as
$$
\tag{5.70}
\hat{y}^{(train)} = X^{(train)}w.
$$
Expressed as a Gaussian conditional distribution on $y^{(train)}$, we have
$$
\begin{eqnarray}
p(y^{(train)} | X^{(train)}, w)
\tag{5.71}
&=& \mathcal{N}(y^{(train)} ; X^{(train)}w, I) \\
\tag{5.72}
&\varpropto& \text{exp} \bigg(
- \frac{1}{2}(y^{(train)} - X^{(train)}w)^{\mathsf{T}} (y^{(train)} - X^{(train)}w)
\bigg),
\end{eqnarray}
$$
where we follow the standard MSE formulation in assuming that the Gaussian variance on $y$ is one.

In what follows, to reduce the notational burden, we refer to ($X^{(train)}, y^{(train)}$) as simply ($X, y$).

To determine the posterior distribution over the model parameter vector $w$, we first need to specify a prior distribution. For real-valued parameters it is common to use a Gaussian as a prior distribution,
$$
\tag{5.73}
p(w) = \mathcal{N}(w; \mu_0, \Lambda_0) \varpropto \text{exp} \bigg(
-\frac{1}{2}(w - \mu_0)^{\mathsf{T}} \Lambda^{-1} (w - \mu_0) \bigg),
$$
where $\mu_0$ and $\Lambda_0$ are the prior distribution mean vector and covariance matrix respectively. (We assume a diagonal covariance matrix $\Lambda_0 = diag(\lambda_0)$, unless there is a reason to use a particular covariance structure.)

With the prior thus specified, we can now proceed in determining the **posterior** distribution over the model parameters:
$$
\begin{eqnarray}
p(w | X, y)
\tag{5.74}
&\varpropto& p(y | X, w)p(w) \\
\tag{5.75}
&\varpropto& 
\text{exp} \bigg(- \frac{1}{2}(y - Xw)^{\mathsf{T}} (y - Xw) \bigg) \text{exp} \bigg(-\frac{1}{2}(w - \mu_0)^{\mathsf{T}} \Lambda^{-1} (w - \mu_0) \bigg) \\
\tag{5.76}
&\varpropto& \text{exp} \bigg(-\frac{1}{2} \big( -2y^{\mathsf{T}}Xw + w^{\mathsf{T}}X^{\mathsf{T}}Xw + w^{\mathsf{T}} \Lambda_0^{-1}w - 2\mu_0^{\mathsf{T}}\Lambda_0^{-1}w \big) \bigg)
\end{eqnarray}
$$
We now define $\Lambda_m = (X^{\mathsf{T}}X + \Lambda_0^{-1})^{-1}$ and $\mu_m = \Lambda_m (X^{\mathsf{T}}y + \Lambda_0^{-1} \mu_0)$. Using these new variables, we find that the posterior may be rewritten as a Gaussian distribution:
$$
\begin{eqnarray}
p(w | X, y)
\tag{5.77}
&\varpropto& 
\text{exp} \bigg(- \frac{1}{2}(w - \mu_m)^{\mathsf{T}} \Lambda_m^{-1}(w - \mu_m) + \frac{1}{2} \mu_m^{\mathsf{T}}\Lambda_m^{-1}\mu_m \bigg) \\
\tag{5.78}
&\varpropto& \text{exp} \bigg(-\frac{1}{2} (w - \mu_m)^{\mathsf{T}} \Lambda_m^{-1}(w - \mu_m) \bigg)
\end{eqnarray}
$$
All terms that do not include the parameter vector $w$ have been omitted; they are implied by the fact that the distribution must be normalized to integrate to 1.

> Equation 3.23 shows how to normalize a multivariate Gaussian distribution:
> $$
> \tag{3.23}
> \mathcal{N}(x; \mu, \Sigma) = \sqrt{\frac{1}{(2\pi)^n \text{det}(\Sigma)}} \text{exp} \bigg(-\frac{1}{2} (x - \mu)^{\mathsf{T}} \Sigma^{-1}(x - \mu) \bigg).
> $$
> When we wish to evaluate the PDF several times for many different values of the parameters, the covariance is not a computationally efficient way to parametrize the distribution, since we need to invert $\Sigma$ to evaluate the PDF. We can instead use a **precision matrix $\beta$**:
> $$
> \tag{3.24}
> \mathcal{N}(x; \mu, \beta^{-1}) = \sqrt{\frac{\text{det}(\beta)}{(2\pi)^n}} \text{exp} \bigg(-\frac{1}{2} (x - \mu)^{\mathsf{T}} \beta (x - \mu) \bigg).
> $$

#### Maximum A Posteriori (MAP) Estimation

> While the most principled approach is to make predictions using the full Bayesian posterior distribution over the parameter $\theta$, it is still often desirable to have a single point estimate. One common reason for desiring a point estimate is that most operations involving the Bayesian posterior for most interesting models are intractable, and a point estimate offers a tractable approximation.
>
> Rather than simply returning to the MLE, we can still gain some of the benefit of the Bayesian approach by allowing the prior to influence the choice of the point estimate. One rational way to do this id to choose the **maximum a posteriori** (MAP) point estimate. The MAP estimate chooses the point of maximal posterior probability ( or maximal probability density in the more common case of continuous $\theta$):
> $$
> \tag{5.79}
> \theta_{MAP}
> = \underset{\mathbf{\theta}}{\operatorname{argmax}} {p(\mathbf{\theta} | x)}
> = \underset{\mathbf{\theta}}{\operatorname{argmax}} {\text{log} \ p(x | \mathbf{\theta})} + \text{log} \ p(\mathbf{\theta})
> $$
> We recognize, on the righthand side, $\text{log} \ p(x | \mathbf{\theta})$, that is, the standard log-likelihood term, and $\text{log} \ p(\mathbf{\theta})$, corresponding to the prior distribution.
>
> As an example, consider a linear regression model with a Gaussian prior on the weights $w$. If this prior is given by $\mathcal{N}(w; 0, \frac{1}{\lambda}I^2)$, then the log-prior term in equation 5.79 is proportional to the familiar $\lambda w^{\mathsf{T}}w$ weight decay penalty, plus a term that does not depend on $w$ and does not affect the learning process. MAP Bayesian inference with a Gaussian prior on the weights thus corresponds to weight decay.



### MLE and MAP: 殊途同归

#### Binary Variables

- Coin flipping: heads = 1, tails = 0 with bias $\mu$
  $$
  p(X = 1 | \mu) = \mu
  $$

- Bernoulli Distribution
  $$
  Bern(x | \mu) = \mu^x \cdot (1 - \mu)^{1 - x} \\
  \mathbf{E}[X] = \mu \\
  var(X) = \mu \cdot (1 - \mu)
  $$

- N coin flips: $X_1, \ldots, X_N$
  $$
  p(\Sigma_i X_i = m | N, \mu) = {N \choose m} \mu^m (1 - \mu)^{N - m} \\
  $$

- Binomial Distribution
  $$
  p(m | N, \mu) = {N \choose m} \mu^m (1 - \mu)^{N - m} \\
  \mathbf{E}[\Sigma_i X_i] = N \mu \\
  var[\Sigma_i X_i] = N \mu (1 - \mu)
  $$

#### The Bias of a Coin

Suppose that we have a coin, and we would like to figure out what the probability is that it will flip up heads.

- How should we estimate the bias?

With these coin flips result: **[tail, head, tail, head, head]**, our estimate of the bias is: 3/5 ("the frequency of heads").

- why is this a good estimate of the bias?

  	- how good is this estimation?

- $P(Heads) = \theta, \ P(Tails) = 1 - \theta$

- Flips are i.i.d.

  - Independent events
  - Identically distributed according to Binomial distribution

- Our training data consists of $\alpha_H$ heads and $\alpha_T$ tails
  $$
  p(D | \theta) = \theta^{\alpha_H} \cdot (1 - \theta)^{\alpha_T}
  $$

#### MLE

- Data: Observed set of $\alpha_H$ heads and $\alpha_T$ tails

- Hypothesis: Coin flips follow a binomial distribution

- Learning: Find the "best" $\theta$

  Maximum Likelihood Estimation: Choose $\theta$ to maximize probability of $D$ given $\theta$

$$
\begin{eqnarray}
\hat{\theta}
&=& \underset{\mathbf{\theta}}{\operatorname{argmax}} \  P(D | \theta) \\
&=& \underset{\mathbf{\theta}}{\operatorname{argmax}} \  \text{ln} \ P(D | \theta) \\
&=& \underset{\mathbf{\theta}}{\operatorname{argmax}} \  \text{ln} \ \theta^{\alpha_H} \cdot (1 - \theta)^{\alpha_T}
\end{eqnarray}
$$

- Set derivative to zero, and solve!

$$
\begin{eqnarray}
\frac{d}{d\theta} \text{ln} \ P(D | \theta)
&=& \frac{d}{d\theta} [\text{ln} \ \theta^{\alpha_H} \cdot (1 - \theta)^{\alpha_T}] \\
&=& \frac{d}{d\theta} [\alpha_H \text{ln} \ \theta + \alpha_T \text{ln} (1 - \theta)] \\
&=& \alpha_H \frac{d}{d\theta} \text{ln} \ \theta + \alpha_T \frac{d}{d\theta} \text{ln} \ (1 - \theta) \\
&=& \frac{\alpha_H}{\theta} - \frac{\alpha_T}{1 - \theta} = 0 \\
\\
\Rightarrow \hat{\theta}_{MLE} &=& \frac{\alpha_H}{\alpha_H + \alpha_T}
\end{eqnarray}
$$

As we can see now, that's exactly the "Frequency of the heads"! In other words, the frequency of heads is exactly the **maximum likelihood estimator** for this problem.

#### MAP

Suppose we have 5 coin flips all of which are heads, Our estimate of the bias is: ???

- MLE would give $\theta_{MLE} = 1$
- This event occurs with probability $1 / 2^5 = 1/32$ for a fair coin
- Are we willing to commit to such a strong conclusion with such little evidence?

**Priors** are a Bayesian mechanism that allow us to take into account "prior" knowledge about our belief in the outcome. Rather than estimating a single $\theta$, consider a distribution over possible values of $\theta$ given the data:

- Without any data observed, our best guess of $\theta$ is obeyed a Beta(2, 2),
- After we see some data (such as observed flips:[tails, tails]), we update our prior to Beta(3, 2).

**Bayesian Learning**
$$
\begin{eqnarray}
\tag{L1}
p(\theta | D) = \frac{p(D | \theta) \ p(\theta)}{p(D)} \\
\tag{L2} \label{eq_map}
\Rightarrow p(\theta | D) \varpropto p(D | \theta) \ p(\theta)
\end{eqnarray}
$$
where

- $p(\theta | D)$ is the posterior,
- $p(D | \theta)$ is the data likelihood,
- $p(\theta)$ is the prior,
- $p(D)$ is the normalization factor.

We update the prior according to the observed data to get the posterior by applying Bayes rule.

**Picking Priors**

How do we pick a good prior distribution? 

- Priors could represent expert domain knowledge
- Statisticians choose them to make the posterior distribution "nice" (conjugate priors, which makes the posterior the same form as the prior)

What is a good prior for the bias in the coin flipping problem?

- Truncated Gaussian (tough to work with)
- Beta distribution (works well for binary random variables)

**Coin Flips with Beta Distribution**

- Likelihood function: $p(D | \theta) = \theta^{\alpha_H} (1 - \theta)^{\alpha_T}$

- Prior: $p(\theta) = \frac{\theta^{\beta_H - 1} (1 - \theta)^{\beta_T - 1}}{B(\beta_H, \beta_T)} \sim Beta(\beta_H, \beta_T)$

- Posterior:
  $$
  \begin{eqnarray}
  p(\theta | D) 
  &\varpropto& \theta^{\alpha_H}(1 - \theta)^{\alpha_T} \theta^{\beta_H - 1}(1 - \theta)^{\beta_T - 1} \\
  &=& \theta^{\alpha_H + \beta_H - 1}(1 - \theta)^{\alpha_T + \beta_T - 1} \\
  &=& Beta(\alpha_H + \beta_H, \alpha_T + \beta_T)
  \end{eqnarray}
  $$

**MAP Estimation**

Choosing $\theta$ to maximize the posterior distribution is called "maximum a posteriori (MAP)" estimation
$$
\theta_{MAP} = \underset{\mathbf{\theta}}{\operatorname{argmax}} \ {p(\mathbf{\theta} | D)}
$$
The only difference between $\theta_{MLE}$ and $\theta_{MAP}$ is that one assumes a **uniform** prior (MLE) and the other allows an arbitrary prior. 

> Recall that:
>
> With uniform prior $p(\theta) \varpropto 1$, according to $\ref{eq_map}$, the posterior $p(\theta | D) \varpropto p(D | \theta)$.

Suppose we have 5 coin flips all of which are heads,

- MLE would give $\theta_{MLE} = 1$

- MLE with a Beta(2, 2) prior gives $\theta_{MAP} = \frac{5 + 2 - 1}{5+2+0+2 - 2} = \frac{6}{7} \approx .857$

- As we see more data, the effect of the prior diminishes
  $$
  \begin{eqnarray}
  \theta_{MAP}
  &=& \frac{\alpha_H + \beta_H - 1}{\alpha_H + \beta_H + \alpha_T + \beta_T - 2} \\
  \\
  &\approx& \frac{\alpha_H}{\alpha_H + \alpha_T} \ (\text{for large number of observations})
  \end{eqnarray} 
  $$

#### Sample Complexity

How many coin flips do we need in order to guarantee that our learned parameter does not differ too much from the true parameter (with high probability)? Say, I want to know the coin parameter $\theta$, within $\epsilon = 0.1$ error with probability at least $1 - \delta = 0.95$.

Using the Chernoff bound, we have
$$
p(|\theta_{true} - \theta_{MLE}| \geq \epsilon) \leq 2e^{-2N \epsilon^2} \\
\delta \geq 2e^{-2N \epsilon^2} \Rightarrow N \geq \frac{1}{2\epsilon^2} \text{ln} \frac{1}{\delta}
$$
