# Time Series Analysis and Forecast

## Statistical methods

Statistical methods developed specifically for time series data.

Univariate time series: AutoRegressive model (AR), ARIMA

Multivariate time series: Vector AutoRegression



### Autoregressive Models (AR models)

> AR is a model that says that future values of a time series are a function of its past values.

The Autoregressive model relies on the intuition that the past predicts the future and so posits a time series process in which the value at a point in time $t$ is a function of the series's values at earlier points in time.

**Using algebra to understand constraints on AR processes** 

It is exactly what its name implies: a regression on past values to predict future values. (Particularly if there is no information other than the time series itself.)

The simplest AR model, an AR(1) model, describes a system as follows:
$$
y_t = b_0 + b_1 \times y_{t-1} + e_t.
$$
The error term is assumed to have a constant variance and a mean of zero[^1]. We denote an autoregressive term that looks back only to the immediately prior time as an AR(1) model because it includes a lookback of one lag.

The generalization of this notation allows the present of an AR process to depend on the $p$ most recent values, producing an AR($p$) process:
$$
y_t = \phi_0 + \phi_1 \times y_{t-1} + \cdots + \phi_p \times y_{t-p} + e_t
$$
The stationarity is a key concept in time series analysis because it is required by many time series models, including AR models.



> **Stationarity**: a stationary time series is one that has fairly stable statistical properties over time, particularly with respect to mean and variance. It is defined as following: a process is stationary if for all possible lags, $k$, the distribution of $y_t, y_{t+1}, \ldots, y_{t+k}$, does not depend on $t$.
>
> Statistical test for stationary often come down to the question of whether there is a unit root -- that is , whether $1$ is a solution of the process's characteristic equation[^2].
>
> A simple intuition for what a unit root is can be gleaned from the example of a random walk:
> $$
> y_t = \phi \times y_{t-1} + e_t
> $$
> In this process, the value of a time series at a given time is a function of its value at the immediately preceding time and some random error. if $\phi$ is equal to $1$, the series has a unit root, will "walk away", and will not be stationary. Random walk also tells us that non-stationary does not imply trending in time series.
>
> **Augmented Dickey-Fuller (ADF) test**: the most commonly used metric to assess a time series for stationarity problems. This test posits a **null** hypothesis that a unit root is present in a time series. And then goes with the *hypothesis tests* procedure. Note that tests for stationarity focus on whether the mean of series is changing. The variance is handled by transformations rather than formally tested.



**...to be continue.** 



[^1]: Gaussian distribution.
[^2]: Determining whether a process has a unit root remains a current area of research.