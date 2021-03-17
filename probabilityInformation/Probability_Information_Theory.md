# Probability and Information Theory

Probability theory is a mathematical framework for representing uncertain statements. It provides a means of quantifying uncertainty as well as axioms (公理) for deriving new uncertain statements. In artificial intelligence applications, we use probability theory in two major ways:

- The laws of probability tell us how AI systems should reason, so we design our algorithms to compute or approximate various expressions derived using probability theory;
- We can use probability and statistics to theoretically analyze the behavior of proposed AI systems.

**While probability theory allows us to make uncertain statements and to reason in the presence of uncertainty, information theory enables us to quantify the amount of uncertainty in a probability distribution.**

## Why Probability?

Machine learning must aways deal with uncertain quantities and sometimes stochastic (nondeterministic) quantities. Uncertainty and stochasticity can arise from many sources. There are three possible sources of uncertainty:

- Inherent stochasticity in the system being modeled. For example, most interpretations of quantum mechanics describe the dynamics of subatomic particles as being probabilistic.
- Incomplete observability. Even deterministic systems can appear stochastic when we cannot observe all the variables that drive the behavior of the system. For example, there are three doors, two doors lead to a goat while a third leads to a car. The show contestant was asked to choose among three doors and wins the prize. The outcome given the contestant's choice is deterministic, but from the contestant's point of view, the outcome is uncertain.
- Incomplete modeling. When we use a model that must discard some of the information we have observed, the discarded information results in uncertainty in the model's predictions.

While it should be clear that we need a means of representing and reasoning about uncertainty, it is not immediately obvious that probability theory can provide all the tools we want for AI applications. Probability theory was originally developed to analyze the frequencies of events. It is easy to see how probability theory can be used to study events like drawing a certain hand of cards in a poker game. These kinds of events are often repeatable. When we say that an outcome has a probability $p$ of occurring, it means that if we repeated the experiment (e.g., drawing a hand of cards, flipping a coin) infinitely many times, then proportion $p$ of the repetitions would result in that outcome. This kind of reasoning does not seem immediately applicable to propositions that are not repeatable. If a doctor analyzes a patient and says that the patient has a 40 percent chance of having the flu, this means something very different -- we cannot make infinitely many replicas of the patient, nor is there any reason to believe that different replicas of the patient would present with the same symptoms yet have varying underlying conditions. In the case of the doctor diagnosing the patient, we use probability to represent a **degree of belief**, with $1$ indicating absolute certainty that the patient has the flu and $0$  indicating absolute certainty that the patient does not have the flu. The former kind of probability, related directly to the rates at which events occur, is known as **frequentist probability**, while the latter, related to qualitative levels of certainty, is known as **Bayesian probability**.

Probability can be seen as the extension of logic to deal with uncertainty. Logic provides a set of formal rules for determining what propositions (命题) are implied to be true or false given the assumption that some other set of propositions is true of false. Probability theory provides a set of formal rules for determining the likelihood of a proposition being true given the likelihood of other propositions.

## Random Variables

A **random variable** is a variable that can take on different values randomly. On its own, a random variable is just a description of the states that are possible; it must be coupled with probability distribution that specifies how likely each of these states are.

*Quantitative* random variables may be discrete or continuous. This is not a hard-and-fast distinction, but it is a useful one. For a discrete variable, the values can only differ by fixed amounts. Family size is discrete. Two families can differ in size by 0 or 1 or 2, and so on. Nothing in between is possible. Age, on the other hand, is a continuous variable. This doesn't refer to the fact that a person is continuously getting older; it just means that the difference in age between two people can be arbitrarily small -- a year, a month, a day, a hour, $\ldots$ And there are variables are *qualitative* : examples are marital status (single, married, widowed, divorced, separated) and employment status (employed, unemployed, not in the labor force). Finally, the terms *qualitative, quantitative, discrete*, and *continuous* are also used to describe data -- qualitative data are collected on a qualitative variable, and so on.

![random variables](./images/stats_random_variables.png)



## Probability Distributions

A **probability distribution** is a description of how likely a random variable or set of random variables is to take on each of its possible states. The way we describe probability distributions depends on whether the variables are discrete or continuous.

### Discrete Variables and Probability Mass Functions (PMF)

A probability distribution over discrete variables may be described using a **Probability Mass Functions (PMF)**. We typically denote probability mass functions with a capital $P$. Often we associate each random variable with different probability mass function and the reader must infer which PMF to use based on the identity of the random variable, rather than on the name of the function; $P(\bf{x})$ is usually not the same as $P(\bf{y})$.

The PMF maps from a state of a random variable to the probability of that random variable taking on that state. The probability that $\mathbf{x} = x$ is denoted explicitly as $P(\mathbf{x}=x)$ or $P(x)$ in brevity, with a probability of $1$ indicating that $\mathbf{x} = x$ is certain and a probability of $0$ indicating that $\mathbf{x} = x$ is impossible. Sometimes we define a variable first, then use $\sim$ notation: $\mathbf{x} \sim P(\mathbf{x})$.

PMF can act on many variables at the same time. Such a probability distribution over many variables is known as a **join probability distribution**. $P(\mathbf{x} = x, \mathbf{y} = y)$ denotes the probability that $\mathbf{x}=x, \mathbf{y}=y$ simultaneously. We may also write $P(x, y)$ for brevity.

To be a PMF on a random variable $\bf{x}$, a function $P$ must satisfy the following properties:

- The domain of $P$ must be the set of all possible states of $\bf{x}$.
- $\forall x \in \mathbf{x}, 0 \le P(x) \le 1$. An impossible event has probability $0$, and no state can be less probable than that. Likewise, an event that is guaranteed to happen has probability $1$, and no state can have a greater chance of occurring.
- $\sum_{x \in \mathbf{x}} P(x) = 1$. We refer to this property as being **normalized**. Without this property, we could obtain probabilities greater than one by computing the probability of one of many events occurring.

For example, consider a single discrete random variable $\bf{x}$ with $k$ different states. We can place a **uniform distribution** on $\bf{x}$ -- that is, make each of its states equally likely -- by setting its PMF to
$$
\tag{3.1}
P(\mathbf{x}=x_i) = {1 \over k}
$$
for all $i$. We can see that this fits the requirements for a probability mass function. The value $1 \over k$ is positive because $k$ is positive integer. We also see that
$$
\tag{3.2}
\sum_i P(\mathbf{x}=x_i) = \sum_i {1 \over k} = {k \over k} = 1,
$$
so the distribution is properly normalized.

### Continuous Variables and Probability Density Functions (PDF)

When working with continuous random variables, we describe probability distributions using a **Probability Density Functions (PDF)** rather than a PMF. To be a Probability Density Function, a function $p$ must satisfy the following properties:

- The domain of $p$ must be the set of all possible states of $\bf{x}$.
- $\forall x \in \mathbf{x}, p(x) \ge 0$. Note that we do not require $p(x) \le 1$.
- $\int p(x)dx = 1$.

**A probability density function $p(x)$ does not give the probability of a specific state directly; instead the probability of landing inside an infinitesimal region with volume $\delta x$ is given by $p(x) \delta x$.** Specifically, the probability that $x$ lies in some set $\mathbb{S}$ is given by the integral of density function $p(x)$ over that set. In the univariate example, the probability that $x$ lies in the interval $[a, b]$ is given by $\int_{[a, b]}p(x)dx$.

For an example of a PDF corresponding to a specific probability density over a continuous random variable, consider a uniform distribution on an interval of the real numbers. We can do this with a function $u(x; a, b)$, where $a$ and $b$ are the endpoints of the interval, with $b > a$. The "$;$" notation means "parametrized by"; we consider $x$ to be the argument of the function, while $a$ and $b$ are parameters that define the function. To ensure that there is no probability mass outside the interval, we say $u(x; a, b) = 0\ \text{for all}\ x \notin [a, b]$. Within $[a, b]$, $u(x; a, b) = {1 \over b-a}$. we can see that this is non-negative everywhere. Additionally, it integrates to $1$. We often denote that $x$ follows the uniform distribution on $[a, b]$ by writing $\mathbf{x} \sim U(a, b)$.

### Marginal Probability

Sometimes we know the probability distribution over a set of variables and we want to know the probability distribution over just a subset of them. The probability distribution over the subset is known as the **marginal probability distribution**.

For example, suppose we have discrete random variables $\bf{x}$ and $\bf{y}$ , and we know $P(\bf{x}, y)$. We can find $P(\bf{x})$ with **sum rule**:
$$
\tag{3.3}
\forall x \in \mathbf{x}, P(\mathbf{x} = x) = \sum_y P(\mathbf{x}=x, \mathbf{y}=y) .
$$
The name "marginal probability" comes from the process of computing marginal probabilities on paper. When the values of $P(\bf{x}, y)$ are written in a grid with different values of $x$ in rows and different values of $y$ in columns, it is natural to sum across a row of the grid, then write $P(\bf{x})$ in the marginal of the paper just to the right of the row (see Table 4.1 from the book [Doing Bayesian Data Analysis]).

![discrete marginal probability](./images/DBDA_marginal_probability.png)

For continuous variables, we need to use integration instead of summation:
$$
\tag{3.4}
p(x) = \int p(x, y)dy .
$$

### Conditional Probability

In many cases, we are interested in the probability of some event, given that some other event has happened. This is called a **conditional probability**. We denote the conditional probability that $\mathbf{y}=y$ given $\mathbf{x}=x$ as $P(\mathbf{y}=y | \mathbf{x}=x)$. This conditional probability can be computed with the formula
$$
\tag{3.5}
P(\mathbf{y}=y | \mathbf{x}=x) = \frac{P(\mathbf{y}=y , \mathbf{x}=x)}{P(\mathbf{x}=x)} .
$$
The conditional probability is only defined when $P(\mathbf{x}=x) > 0$. We cannot compute the conditional probability conditioned on an event that never happens.

It is important not to confuse conditional probability with computing what would happen if some action were undertaken. The conditional probability that a person is from Germany given that they speak German is quite high, but if a randomly selected person is taught to speak German, their country of origin does not change. Computing the consequences of an action is called making an **intervention query**. Intervention queries are the domain of **causal modeling**.

### The Chain Rule of Conditional Probability

Any join probability distribution over many random variables may be decomposed into conditional distribution over only one variable:
$$
\tag{3.6}
P(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}) = P(\mathbf{x}^{(1)}) \prod^n_{i=2} P(\mathbf{x}^{(i)} | \mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(i-1)}) .
$$
This observation is known as the **chain rule**, or **product rule**, of probability. It follows immediately from the definition of conditional probability in equation 3.5.

For example, applying the definition twice, we get
$$
\begin{eqnarray}
P(a, b, c) &=& P(a | b, c) P(b, c) \\
\\
P(b, c) &=& P(b | c) P(c) \\
\\
P(a, b, c) &=& P(a | b, c) P(b | c) P(c) .
\end{eqnarray}
$$

### Independence and Conditional Independence

Two random variables $\bf{x}$ and $\bf{y}$ are **independent** if their probability distribution can be expressed as a product of two factors, one involving only  $\bf{x}$ and one involving only $\bf{y}$ :
$$
\tag{3.7}
\forall x \in \mathbf{x}, y \in\mathbf{y}, p(\mathbf{x}=x, \mathbf{y}=y) = p(\mathbf{x}=x) p(\mathbf{y}=y) .
$$
Two random variables $\bf{x}$ and $\bf{y}$ are **conditional independent** given a random variable $z$ if the conditional probability distribution over $\bf{x}$ and $\bf{y}$ factorizes in this way for every value of $z$ :
$$
\tag{3.8}
\forall x \in \mathbf{x}, y \in\mathbf{y}, z \in \mathbf{z},
p(\mathbf{x}=x, \mathbf{y}=y | \mathbf{z}=z) = 
p(\mathbf{x}=x | \mathbf{z}=z) p(\mathbf{y}=y | \mathbf{z}=z) .
$$
We can denote  **independent** and **conditional independent** with compact notation:

-  $\bf{x} \perp y$ means that  $\bf{x}$ and $\bf{y}$ are independent,
-  $\bf{x} \perp y | z$ means that  $\bf{x}$ and $\bf{y}$ are conditional independent given $z$.

### Expectation, Variance and Covariance

The **expectation**, or **expected value**, of some function $f(x)$ with respect to a probability distribution $P(\mathbf{x})$ is the average, or mean value, that $f$ takes on when $x$ is drawn from $P$. For discrete variables this can be computed with summation:
$$
\tag{3.9}
\mathbb{E}_{\mathbf{x} \sim P} [f(x)] = \sum_x P(x)f(x),
$$
while for continuous variables, it is computed with an integral:
$$
\tag{3.10}
\mathbb{E}_{\mathbf{x} \sim P} [f(x)] = \int p(x)f(x)dx,
$$
Expectations are linear, for example,
$$
\tag{3.11}
\mathbb{E}_{\mathbf{x}} [\alpha f(x) + \beta g(x)] =
\alpha \mathbb{E}_{\mathbf{x}}[f(x)] + \beta \mathbb{E}_{\mathbf{x}}[g(x)] ,
$$
when $\alpha$ and $\beta$ are not dependent on $x$.

The **variance** gives a measure of how much the values of a function of a random variable $\bf{x}$ vary as we sample different values of $x$ from its probability distribution:
$$
\tag{3.12}
\text{Var}(f(x)) = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^2] .
$$
When the variance is low, the values of $f(x)$ cluster near their expected value. The square root of the variance is known as the **standard deviation**.

The **covariance** gives some sense of how much two values are linearly related to each other, as well as the scale of these variables:
$$
\tag{3.13}
\text{Cov}(f(x), g(y)) = \mathbb{E} \bigg[\big(
f(x) - \mathbb{E}[f(x)])\ (g(y) - \mathbb{E}[g(y)]
\big)\bigg] .
$$
High absolute values of the covariance mean that the values change very much and are both far from their respective means at the same time.

- If the sign of the covariance is positive, both variables tend to take on relatively high values simultaneously,
- If the sign of the covariance is negative, one variable tend to take on relatively high values and the other takes on a relatively low values and vice versa.

The notions of covariance and dependence are related but distinct concepts.

- Two independent variables have zero covariance, and two variables have nonzero covariance are dependent.
- Zero covariance means there must be no linear dependence, While independent also excludes nonlinear relationships between two variables.
- It is possible for two variables to be dependent but have zero covariance.

The **covariance matrix** of a random vector $x \in \R^n$ is an $n \times n$ matrix, such that
$$
\tag{3.14}
\text{Cov}(\mathbf{x})_{i, j} = \text{Cov}(\mathbf{x}_i, \mathbf{x}_j) .
$$
The diagonal elements of the covariance  give the variance:
$$
\tag{3.15}
\text{Cov}(\mathbf{x}_i, \mathbf{x}_i) = \text{Var}(\mathbf{x}_i) .
$$


### Common Probability Distribution

#### Bernoulli Distribution

The Bernoulli Distribution is a distribution over a single binary random variable. It is controlled by a single parameter $\phi \in [0, 1]$, which gives the probability of the random variable being equal to $1$. It has the following properties:
$$
\begin{eqnarray}
\tag{3.16}
P(\mathbf{x} = 1) &=& \phi \\
\\
\tag{3.17}
P(\mathbf{x} = 0) &=& 1 - \phi \\
\\
\tag{3.18}
P(\mathbf{x} = x) &=& \phi^{x} (1 - \phi)^{1-x} \\
\\
\tag{3.19}
\mathbb{E}_\mathbf{x}[\mathbf{x}] &=& \phi \\
\\
\tag{3.20}
\text{Var}_\mathbf{x}(\mathbf{x}) &=& \phi (1 - \phi)
\end{eqnarray}
$$


#### Gaussian Distribution

The most commonly used distribution over real numbers is the normal distribution, also known as the Gaussian distribution:
$$
\tag{3.21}
\mathcal{N}(x; \mu, \sigma^2) = \sqrt{{1 \over {2 \pi \sigma^2}}} \exp \bigg(-{1 \over {2 \sigma^2}} (x - \mu)^2 \bigg).
$$
The two parameters $\mu \in \R$ and $\sigma \in (0, \infin)$ control the normal distribution.

When we evaluate the PDF, we need to square and invert $\sigma$. When we need to frequently evaluate the PDF with different parameter values, a more efficient way of parametrizing the distribution is to use a parameter $\beta \in (0, \infin)$ to control the **precision**, or inverse variance, of the distribution:
$$
\tag{3.22}
\mathcal{N}(x; \mu, \beta^{-1}) = \sqrt{{\beta \over {2 \pi}}} \exp \bigg(-{1 \over 2} \beta (x - \mu)^2 \bigg).
$$
![bell curve](./images/DL_gaussian_distribution.png)

Normal distributions are a sensible choice for many applications, for two major reasons:

- First, many distributions we wish to model are truly close to being normal distributions. The **central limit theorem** shows that the sum of many independent random variables is approximately normally distributed. This means that in practice, many complicated systems can be modeled successfully as normally distributed noise, even if the system can be decomposed into parts with more structured behavior.
- Second, out of all possible probability distributions with the same variance, the normal distribution encodes the maximum amount of uncertainty over the real numbers. We can thus think of the normal distribution as being the one that inserts the least amount of prior knowledge into a model.

The normal distribution generalizes to $\R^n$, in which case it is known as the **multivariate normal distribution**. It may be parametrized with a positive definite symmetric matrix $\Sigma$ :
$$
\tag{3.23}
\mathcal{N}(x; \mu, \Sigma) = \sqrt{{1 \over {(2 \pi)^n |\Sigma|}}} 
\exp \bigg(-{1 \over 2} (x - \mu)^{\mathsf{T}} \Sigma^{-1} (x - \mu) \bigg).
$$
The parameter $\mu$ still gives the mean of the distribution, though now it is vector valued. The parameter $\Sigma$ gives the covariance matrix of the distribution. $|\Sigma|$ denotes the **determinant** of $\Sigma$ .

As in the univariate case, when we wish to evaluate the PDF multiple times with more efficient computation, we need to invert  $\Sigma$ , and instead use a **precision matrix $\beta$** :
$$
\tag{3.24}
\mathcal{N}(x; \mu, \beta^{-1}) = \sqrt{{|\beta| \over (2 \pi)^n}} \exp \bigg(-{1 \over 2} (x - \mu)^{\mathsf{T}} \beta (x - \mu) \bigg).
$$
We often fix the covariance matrix to be a diagonal matrix. An even simpler version is the **isotropic** Gaussian distribution, whose covariance matrix is a scalar times the identity matrix.

#### Exponential and Laplace Distributions

In the context of deep learning, we often want to have a probability distribution with a sharp point at $x = 0$. To accomplish this, we can use the **exponential distribution**:
$$
\tag{3.25}
p(x; \lambda) = \lambda \mathbf{1}_{x \ge 0} \exp(- \lambda x) .
$$
The exponential distribution uses the **indicator function** $\mathbf{1}_{x \ge 0}$ to assign probability zero to all negative values of $x$.

A closely related probability distribution that allows us to place a sharp peak of probability mass at an arbitrary point $\mu$ is the **Laplace distribution**
$$
\tag{3.26}
\text{Laplace}(x; \mu, \gamma) = {1 \over 2\gamma} \exp(- {|x - \mu| \over \gamma}).
$$

## Information Theory

Information theory is a branch of applied mathematics that resolves around quantifying how much information is present in a signal. In the context of machine learning, we can apply information theory to characterize probability distributions or quantify similarity between probability distributions.

The basic intuition behind information theory is that learning that an unlikely event has has occurred is more informative than learning that a likely event has occurred. A message saying "the sun rose this morning" is so uninformative (to human being on the earth) as to be unnecessary to send, but a message saying "there are a solar eclipse this morning" is very informative.

We would like to quantify information in a way that formalizes this intuition.

- Likely events should have low information content, and in the extreme case, events that are guaranteed to happen should have no information content whatsoever.
- Less likely events should have higher information content.
- Independent events should have additive information. For example, finding out that a tossed coin has come up as heads twice should convey twice as much information as finding out that a tossed coin has come up as head once. 

To satisfy all three of these properties, we define the **self-information** of an event $\mathbf{x} = x$ to be
$$
\tag{3.48}
I(x) = - \log P(x) .
$$
In this series, we always use log to mean the natural logarithm, with base $e$. Our definition of $I(x)$ is therefore written in units of **nats**. One nat is the amount of information gained by observing an event of probability $1 \over e$. Other texts use base-2 logarithms and units called **bits** or **shannons**; information measured in bits is just a rescaling of information measured in nats.

When $\bf{x}$ is continuous, an event with unit density still has zero information, despite not being an event that is guaranteed to occur.

### Entropy

Self-information deals only with a single outcome. We can quantify the amount of uncertainty in an entire probability distribution using the **Shannon entropy**,
$$
\tag{3.49}
H(\mathbf{x})
= \mathbb{E}_{\mathbf{x} \sim P}[I(x)]
= - \mathbb{E}_{\mathbf{x} \sim P}[\log P(x)],
$$
also denoted $H(P)$. In other words, the Shannon entropy of a distribution is the expected amount of information in an event drawn from that distribution. It gives a lower bound on the number of nats needed on average to encode symbols drawn from a distribution $P$. Distributions that are nearly deterministic (where outcome is nearly certain) have low entropy; distributions that are closer to uniform have high entropy. When $\bf{x}$ is continuous, the Shannon entropy is known as the **differential entropy**.

![Shannon entropy](./images/DL_binary_Shannon_entropy.png)



### KL divergence

If we have two separate probability distributions $P(x)$ and $Q(x)$ over the same random variable $\bf{x}$, we can measure how different these two distributions are using the **Kullback-Leibler (KL) divergence**:
$$
\tag{3.50} \label{eq_kld}
D_{KL}(P || Q)
= \mathbb{E}_{\mathbf{x} \sim P} \bigg[\log {P(x) \over Q(x)} \bigg]
= \mathbb{E}_{\mathbf{x} \sim P}[\log P(x) - \log Q(x)] .
$$

In the case of discrete variables, it is the extra amount of information (in bits or nats) needed to send a message containing symbols drawn from probability distribution $P$, when we use a code that was designed to minimize the length of messages drawn from probability distribution $Q$.

The KL divergence has many useful properties, most notably being non-negative. The KL divergence is $0$ if and only if $P$ and $Q$ are the same distribution in the case of discrete variables, or equal "almost everywhere[^1]" in the case of continuous variables. Because the KL divergence is non-negative and measures the difference between two distributions, it is often conceptualized as measuring some sort of distance between these distributions. It is not a true distance measure because it is not symmetric: $D_{KL}(P || Q) \ne D_{KL}(Q || P)$ for some $P$ and $Q$. This asymmetriy means that there are important consequences to the choice of whether to use $D_{KL}(P || Q)$ or $D_{KL}(Q || P)$. See figure 3.6 for more detail.

![KL divergence](./images/DL_kl_divergence.png)

A quantity that is closely related to the KL divergence is the **cross-entropy** $H(P, Q) = H(P) + D_{KL}(P||Q)$, which is similar to the KL divergence ($\ref{eq_kld}$) but lacking the term on the left:
$$
\tag{3.51}
H(P, Q) = - \mathbb{E}_{\mathbf{x} \sim P} \log Q(x) .
$$
Minimizing the cross-entropy with respect to $Q$ is equivalent to minimizing the KL divergence, because $Q$ does not participate in the omitted term.

When computing many of these quantities, it is common to encounter expressions of the form $0\log0$. By convention, in the context of information theory, we treat these expressions as $\lim_{x \rightarrow 0} x \log x = 0$.

[^1]: technical view of continuous variables [more advance mathematics knowledge required].

### KL 散度

KL 散度，也称为相对熵(relative entropy)或信息散度(information divergence)，可用于度量两个概率分布之间的差异。给定两个(连续型)概率分布 $P$ 和 $Q$ ，二者之间的 KL 散度定义为
$$
\tag{C.34}
KL(P||Q) = \int^{\infin}_{-\infin} p(x) \log {p(x) \over q(x)} dx,
$$
其中，$p(x)$ 和 $q(x)$ 分别为 $P$ 和 $Q$ 的概率密度函数。KL 散度满足非负性，但不满足对称性，因此 KL 散度不是一个度量(metric)[^2]。

若将 KL 散度的定义展开，可得
$$
\begin{eqnarray}
KL(P||Q) 
&=& \int^{\infin}_{-\infin} p(x) \log {p(x)} dx - \int^{\infin}_{-\infin} p(x) \log {q(x)} dx \\
\\
\tag{C.37}
&=& -H(P) + H(P,Q)
\end{eqnarray}
$$
其中，$H(P)$ 为熵(entropy)，$H(P,Q)$ 为 $P$ 和 $Q$ 得交叉熵(cross-entropy)。在信息论中，熵 $H(P)$ 表示对来自 $P$ 的随机变量进行编码所需要的最小字节数，而交叉熵 $H(H,Q)$ 则表示使用基于 $Q$ 的编码对来自 $P$ 的变量进行编码所需要的 “额外” 的字节数；显然，额外字节数必然非负，当且仅当 $P = Q$ 时额外字节数为零。

[^2]: 度量(metric)应满足四个基本性质：非负性($dist(x_i,x_j) \ge 0$)；同一性($dist(x_i, x_j)=0 \iff x_i=x_j$)；对称性($dist(x_i, x_j) = dist(x_j, x_i)$)；直递性($dist(x_i, x_j) \le dist(x_i, x_k) + dist(x_k, x_j)$)



## Density Estimation

### Histogram

> "In a (standard) histogram, the areas of the blocks represent percentages."
>
> "With the density scale on the vertical axis, the areas of the blocks come out in percent. The area under the histogram over an interval equals the percentage of cases in that interval. The total area under the histogram is 100%."

A **histogram** is a plot designed to show the distribution of values in a set of data. The values are first sorted, and then divided into a fixed number of equal-width bins. A plot is then drawn that shows the number of elements in each bin. (Note that the histograms in Figure 15.20 were not turn the counts into normalized probability density.)

![histograms](./images/ICPP_histograms_coin_flips.png)

A histogram is a depiction of a **frequency distribution**. It tells us how often a random variable has taken on a value in some range, e.g., how often the fraction of times a coin came up heads was between 0.4 and 0.5. It also provides information about the relative frequency of various ranges. For example, we can easily see that the fraction of heads falls between 0.4 and 0.5 far more frequently than it falls between 0.3 and 0.4 (see the one in the left in Figure 15.20).  Notice that while the means in both plots (Figure 15.20) are about the same, the standard deviations are quite different. The spread of outcomes is much tighter when we flip the coin 1000 times per trail than when we flip the coin 100 times per trial[^3].

A **probability distribution** captures the notion of relative frequency by giving the probability of a random value taking on a value within a range. Probability distributions fall into two groups: discrete probability distributions and continuous probability distributions, depending upon whether they define the probability distribution for a discrete or a continuous random variable.

**Discrete probability distributions** are easier to describe. Since there are a finite number of values that the variable can take on, the distribution can be described by simply listing the probability of each value.

**Continuous probability distributions** are trickier. Since there are an infinite number of possible values, the probability that a continuous random variable will take on a specific value is usually 0. For example, the probability that a car is traveling at exactly 81.3457283 miles per hour is probably 0. Mathematicians like to describe continuous probability distributions using a **Probability Density Function (PDF)**. A **PDF** describes the probability of a random variable lying between two values. Think of the PDF as defining a curve where the values on the x-axis lie between the minimum and maximum value of the random variable. Under the assumption that $x_1$ and $x_2$ lie in the domain of the random variable, the probability of the variable having a value between $x_1$ and $x_2$ is the area under the PDF curve between $x_1$ and $x_2$ .

![pdf](./images/ICPP_probability_density_functions.png)

The `random.random()` returns a value lies in interval [0, 1]. The area under the curve of PDF for `random.random()` from 0 to 1 is 1. On the other hand, if we consider the area under the part of the curve between 0.2 and 0.4, it is 0.2 ($(0.4 - 0.2) \times 1.00$). Similarly, the area under the curve for `random.random()+random.random()` from 0 to 2 is also 1, and the area under the curve from 0 to 1 is 0.5. Notice that the same length of interval has the same probability in PDF for `random.random()` while some intervals are more probable than others in PDF for `random.random()+random.random()`.



[^3]: Recall that this is also an example of the law of the average.





### Histogram for density estimation

Density estimation can be used of probability distributions (such Gaussian, Beta, Dirichlet distributions) having specific functional forms governed by a small number of parameters whose values are to be determined from a data set. This is called the *parametric* approach to density modeling. An important limitation of this approach is that the chosen density might be a poor model of the distribution that generates the data, which can result in poor predictive performance. For instance, if the process that generates the data is multimodal (多峰的), then this aspect of the distribution can never be captured by a Gaussian, which is necessarily unimodal[^4].

We consider some *nonparametric* approaches to density estimation that make few assumptions about the form of the distribution. Here we focus mainly on simple frequentist methods. Let's start with a discussion of histogram methods for density estimation, we explore the properties of histogram density model, focusing on the case of a single continuous variable $x$.

Standard histograms simply partition $x$ into distinct bins of with $\Delta_i$ and then count the number $n_i$ of observations of $x$ falling in bin $i$. In order to turn this count into a normalized probability density, we simply divide by the total number $N$ of observations and by the width $\Delta_i$ of the bins to obtain probability values for each bin given by
$$
\tag{2.241}
p_i = \frac{n_i}{N \Delta_i}
$$
for which it is easily seen that $\int p(x)dx = 1$. This gives a model for the density $p(x)$ that is constant over the width of each bin, and often the bins are chosen to have the same width $\Delta_i = \Delta$.

![histogram models](./images/PRML_histogram_models.png)

From above figure, we can see that

- when $\Delta$ is very small (top right), the resulting density model is very spiky (lost a lot of structure of underlying distribution),
- if  $\Delta$ is too large (bottom right), the result is a model that is too smooth (fails to capture the bimodal property of the green curve).

The best results are obtained for some intermediate value of $\Delta$ (middle right). In principle, a histogram density model is also dependent on the choice of edge location for the bins, though this is typically much less significant than the value of $\Delta$.

Note that the histogram method has the property that once the histogram has been computed, the data set itself can be discarded, which can be advantageous if the data set is large. Also, the histogram approach is easily applied if the data points are arriving sequentially.

In practice, the histogram technique can be useful for obtaining a quick visualization of data in one or two dimensions but is unsuited to most density estimation applications. The limitations are

- the estimated density has discontinuities that are due to the bin edges, 
- in a space of high dimensionality, the number of bins (M) are exponential scaling with the number of dimension (D), which is $M^D$. The quantity of data needed to provide meaningful estimates of local probability density would be prohibitive.

Two widely used nonparametric techniques for density estimation, **kernel estimators** and **nearest neighbors**, which have better scaling with dimensionality than the simple histogram model.



[^4]: Consider Gaussian Mixture Models in this case.















