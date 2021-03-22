[toc]



## The regression line

### The method of Least Squares

Sometimes the points on a scatter diagram seem to be following a line. The problem discussed in this section is **how to find the line which best fits the points**. Usually, this involves a compromise: moving the line closer to some points will increase it distance from others. To resolve the conflict, two steps are necessary.

- First, define an average distance from the line to all the points.
- Second, move the line around until this average distance is as small as possible.

To be more specific, suppose the line will be used to predict $y$ from $x$. Then the error made at each point is the vertical distance from the point to the line. In statistics, the usual way to define the average distance is by taking the root-mean-square of the errors. This measure of average distance is called the *r.m.s error of the line*. (It was first proposed by Gauss)

The second problem, how to move the line around to minimize the r.m.s error, was also solved by Gauss.

---

Among all lines, the one that makes the smallest r.m.s error in predicting $y$ form $x$ is the **regression line**.

---

> Recall that:
>
> The r.m.s error for regression says how far typical points are above or below the regression line.
> $$
> r.m.s\ error = \sqrt{\frac{1}{m} \sum^m_i (y_i - \hat{y_i})^2}
> $$
> where $m$ is the number of data points, $y_i$ the $i$-th actual value, $\hat{y_i}$ the corresponding predicted value.
>
> And the r.m.s error for the regression line of $y$ on $x$ can be figured as
> $$
> \sqrt{1 - r^2} \times SD_y
> $$
> where $r$ is the correlation coefficient between $x$ and $y$.

For this reason, the regression line is often called *least squares line*: the errors are squared to compute the r.m.s error, and the regression line makes the r.m.s error as small as possible.

$\color{Green}{ \text{Example}}$

According to Hooke's law, the amout of stretch is proportional to the weight $x$. The new length of the spring is
$$
y = mx + b.
$$
In this equation, $m$ and $b$ are constants which depend on the spring. Their values are unknown, and have to be estimated using experimental data.



Table 1. Data on Hooke's law.

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

Our goal is to estimate $\hat{m}$ and $_\hat{b}$ in the equation of Hooke's law for this spring:
$$
y = \hat{m} x + \hat{b}
$$
The graph of this equation is a perfect straight line. If the points in figure 5 happened to fall exactly on some line, the slope[^2] of that line would estimate $m_1$, and its intercept would estimate $b_1$. However, the points do not line up perfectly. Many different lines could be drawn across the scatter diagram, each having a slightly different slope and intercept.

Which line should be used? Hooke's equation predicts the length from weight. As discussed above, it is natural to choose $m$ and $b$ so as to minimize the r.m.s error, the line $y = \hat{m} x + \hat{b}$ which does the job is the **regression line**. This is the *method of least squares*. In other words, $m$ in Hooke's law should be estimated as the slope of the regression line, and $b$ as its intercept. These are called *least squares estimate*, because they minimize root-mean-square error.

Let's do the arithmetic (in python code):

```python
import numpy as np

# X the weight data; y the length data
X = np.array([0, 2, 4, 6, 8, 10])
y = np.array([439.00, 439.12, 439.21, 439.31, 439.40, 439.50])
# mean and SD
mu_x = X.mean()
mu_y = y.mean()
print(f"The means of X and y: {mu_x, mu_y}")
# The means of X and y: (5.0, 439.25666666666666)
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

# the intercept, this is the predicted length when weight is 0,
# in other words, it should be 5 kg below average.
b_hat = mu_y - (mu_x * m)
# b_hat = 439.0109523809524
```

this gives us: $\hat{m} \approx 0.05$ per kg and $\hat{b} \approx 439.01$ cm.

The length of the spring under no load is estimated as 439.01 cm. And each kilogram of load causes the spring to stretch by about 0.05 cm. Of course, even Hooke's law has its limits: beyond some point, the spring will break. **Extrapolating beyond the data is risky**.

The method of least squares and the regression method involve the same mathematics; but the contexts may be different. In some fields, investigators talk about "least squares" when they are estimating parameters -- unknown constants of nature like $m$ and $b$ in Hooke's law. In other fields, investigators talk about regression when they are studying the relationship between two variables, like income and education, using non-experimental data.

**A technical point:** The least squares estimate for the length of the spring under no load was 439.01 cm. This is a tiny bit longer than the measured length at no load (439.00 cm). A statistician might trust the least squares estimate over the measurement. Why? Because the least squares estimate takes advantage of all six measurements, not just once: some of the measurement error is likely to cancel out. Of course, the six measurements are tied together by a good theory -- Hooke’s law. Without the theory, the least squares estimate wouldn’t be worth much.





[^1]: Convert each variable to standard units. The average of the products gives the correlation coefficient.
[^2]: Associated with a unit increase in $x$ there is some average change in $y$. The slope of the regression line estimates this change. The formula for the slope is $\frac{r \times SD_y}{SD_x}$. And the intercept of the regression line is just the predicted value for $y$ when $x$ is $0$.



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
2.  Do all the predictors help to explain $Y$, or is only a subset of the predictors useful?
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

> $\color{Red}{TODO:}$ this need a separate section to tell the whole story.
>
> **About p-value and the normalization approximation**
>
> The

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

#### Four: Prediction

Once we have fit the multiple regression model, it is straightforward to apply the fitted model $\hat{y} = \hat{f}(X) = \hat{\beta} X$ (a more verbose version see $\ref{mlrpred}$) in order to predict the response based on the values of the predictors. However, there are three sorts of uncertainty associated with this prediction.

1. The coefficient estimate is the least squares estimation of the true coefficient which is unknown. The inaccuracy in the coefficient estimates is related to the *reducible error*[^9]. We can compute a **confidence interval** in order to determine how close $\hat{y}$ will be to $f(X)$.
2. In practice assuming a linear model for $f(X)$ is almost always an approximation of reality, so if the true pattern is non-linear, there is an additional reducible error called *model bias*.
3. Even if we knew $f(X)$ -- that is, we knew the true value of $\beta$ -- the response value cannot be predicted perfectly, because of the random error $\epsilon$ in the model ($\ref{mlrpred}$), this is the *irreducible error*.

### Summary and Overview



















