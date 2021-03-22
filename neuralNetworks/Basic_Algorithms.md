# 8.3 Basic Algorithms

Most deep learning algorithms involve optimization of some sort. Optimization refers to the task of either minimizing or maximizing some function $f(x)$ by altering $x$. The function we want to minimize or maximize is called the **objective function**, or **criterion**. When we are minimizing it, we may also call it the **cost function**, **loss function**, or **error function**.

Suppose we have a function $y = f(x)$, where both $x$ and $y$ are real numbers. The **derivative** of this function is denoted as $f'(x)$ or as $\frac{dx}{dy}$. The derivative $f'(x)$ gives the slope of $f(x)$ at the point $x$. In other words, it specifies how to scale a small change in the input to obtain the corresponding change in the output: $f(x + \epsilon) \approx f(x) + \epsilon f'(x)$.

The derivative is therefore useful for minimizing a function because it tells us how to change x in order to make a small improvement in $y$. For example, we know that $f(x - \epsilon \ \text{sign}(f'(x)))$  is less than $f(x)$ for small enough $\epsilon$. We can thus reduce $f(x)$ by moving $x$ in small steps with opposite sign of the derivative. This technique is called **gradient descent**.

![derivative](./images/gradient_descent_using_derivative.png)

> Note that:
>
> **6.5 Back-Propagation and Other Differentiation Algorithms**
>
> When we use a feedforward neural network to accept an input $x$ and produce an output $\hat{y}$, information flows forward through the network. The input $x$ provides the initial information that then propagates up to the hidden units at each layer and finally produces $\hat{y}$. This is called **forward propagation**. During training, forward propagation can continue onward until it produces a scalar cost $J(\theta)$. The **back-propagation** algorithm, often simply called **backprop**, allows the information from the cost to then flow backward through the network in order to compute the gradient.
>
> Computing an analytical expression for the gradient is straightforward, but numerically evaluating such an expression can be computationally expensive. The back-propagation algorithm does so using a simple and inexpensive procedure.
>
> The term back-propagation is often misunderstood as meaning the whole learning algorithm for multi layer neural networks. Actually, back-propagation refers only to the method for computing the gradient, while another algorithm, such SGD, is used to perform learning using this gradient.
>
> Furthermore, back-propagation is often misunderstood as being specific to multilayer neural networks, but in principle it can compute derivatives of any function ( for some function, the correct response is to report that the derivative of the function is undefined).
>
> Specifically, we will describe how to compute the gradient $\nabla_{x} f(x, y)$ for an arbitrary function $f$, where $x$ is a set of variables whose derivatives are desired, and $y$ is an additional set of variables that are inputs to the function but whose derivatives are not required. In learning algorithms, the gradient we most often require is the gradient of the cost function with respect to the parameters, $\nabla_{\theta} J(\theta)$.



## 8.3.1 Stochastic Gradient Descent

SGD and its variants are probably the most used optimization algorithms for machine learning in general and for deep learning in particular. It is possible to obtain an unbiased estimate of the gradient by taking the average gradient on a minibatch of $m$ examples drawn i.i.d from the data-generating distribution.

Algorithm 8.1 shows how to follow this estimate of the gradient downhill.

***

**Algorithm 8.1** Stochastic gradient descent (SGD) update at training iteration $k$

***

**Require:** Learning rate $\epsilon_k$

**Require:** Initial parameter $\theta$

​	**while** stopping criterion not met **do**

​		Sample a minibatch of $m$ examples from the training set {$x^{(1)}, \ldots, x^{(m)}$} with corresponding targets $y^{(i)}$

​		Compute gradient estimate: $\hat{g} \leftarrow + \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})$

​		Apply update: $\theta \leftarrow \theta - \epsilon \hat{g}$

​	**end while**

***

> Recall that:
>
> The cost function used by a machine learning algorithm often decomposes as a sum over training examples of some per-example loss function. For example, the negative conditional log-likelihood of the training data can be written as 
> $$
> \tag{5.96}
> J(\theta) = \mathbb{E}_{\mathbf{x}, \mathbf{y} \sim \hat{P}_{data}} L(x, y, \theta) = \frac{1}{m} \sum^m_{i=1} L(x^{(i)}, y^{(i)}, \theta),
> $$
> where $L$ is the per-example loss $L(x, y, \theta) = - log \ p(y | x; \theta)$.
>
> $\color{Blue}{\textsf{For these additive cost functions, gradient descent requires computing}}$
> $$
> \tag{5.97}
> \nabla_{\theta} J(\theta) = \frac{1}{m} \sum^m_{i=1} \nabla_{\theta} L(x^{(i)}, y^{(i)}, \theta)
> $$
> The computational cost of this operation is $O(m)$. As the training set size grows to billions of examples, the time to take a single gradient step becomes prohibitively long.
>
> The insight of SGD is that the gradient is an expectation. The expectation may be approximately estimated using a small set of samples. Specifically, on each step of the algorithm, we can sample a minibatch of $m'$ examples, then the estimate of the gradient is formed as
> $$
> \tag{5.98}
> \hat{g} = \frac{1}{m'} \sum^{m'}_{i=1} \nabla_{\theta} L(x^{(i)}, y^{(i)}, \theta)
> $$

A crucial parameter for the SGD algorithm is the learning rate. In practice, it is necessary that SGD algorithm to gradually decrease the learning rate over time (Batch gradient descent can use a fixed learning rate), this is because the SGD gradient estimator introduces a source of noise (the random sampling of $m$ training examples) that does not vanish even when we arrive at a minimum. A sufficient condition to guarantee convergence of SGD is that
$$
\tag{8.12}
\sum^{\infin}_{k=1} \epsilon_k = \infin, \ \text{and}
$$

$$
\tag{8.13}
\sum^{\infin}_{k=1} \epsilon^2_k < \infin.
$$

In practice, it is common to decay the learning rate linearly until iteration $\tau$:
$$
\tag{8.14}
\epsilon_k = (1 - \alpha)\epsilon_0 + \alpha \epsilon_{\tau}
$$
with $\alpha = \frac{k}{\tau}$. After iteration $\tau$, it is common to leave $\epsilon$ constant.



## 8.3.2 Momentum

While SGD remains a popular optimization strategy, learning with it can sometimes be slow. The method of momentum is designed to accelerate learning, especially in the face of high curvature, small but consistent gradients, or noisy gradients. The momentum algorithm accumulates an exponentially decaying moving average of past gradients and continues to move in their direction.

Formally, the momentum algorithm introduces a variable $v$ that plays the role of velocity -- it is the direction and speed at which the parameters  move through parameter space. The velocity is set to an exponentially decaying average of the negative gradient.

In the momentum learning algorithm, we assume unit mass, so the velocity vector $v$ may also be regarded as the momentum of the particle (physical analogy, Newton's laws of motion). A hyperparameter $\alpha \in [0, 1)$ determines how quickly the contributions of previous gradients exponentially decay. The update rule is given by
$$
\begin{eqnarray}
\tag{8.15}
v &\leftarrow& \alpha v - \epsilon \nabla_{\theta} \bigg(\frac{1}{m} \sum^m_{i=1} L(f(x^{(i)}; \theta), y^{(i)}) \bigg), \\

\tag{8.16}
\theta &\leftarrow& \theta + v
\end{eqnarray}
$$

***

**Algorithm 8.2** Stochastic gradient descent (SGD) with momentum

***

**Require:** Learning rate $\epsilon$, momentum parameter $\alpha$

**Require:** Initial parameter $\theta$, initial velocity $v$

​	**while** stopping criterion not met **do**

​		Sample a minibatch of $m$ examples from the training set {$x^{(1)}, \ldots, x^{(m)}$} with corresponding targets $y^{(i)}$

​		Compute gradient estimate: $\hat{g} \leftarrow + \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})$

​		Compute velocity update: $v \leftarrow \alpha v - \epsilon \hat{g}$

​		Apply update: $\theta \leftarrow \theta + v$

​	**end while**

***

The velocity $v$ accumulates the gradient elements $\nabla_{\theta} (\frac{1}{m} \sum^m_i L(f(x^{(i)}; \theta), y^{(i)}))$. The larger $\alpha$ is relative to $\epsilon$, the more previous gradients affect the current direction.

Previously, the size of the step was simply the norm of the gradient multiplied by the learning rate. Now, the size of the step depends on how large and how aligned a *sequence* of gradients are. The step size is largest when many successive gradients point in exactly the same direction. If the momentum algorithm always observes gradient $g$, then it will accelerate in the direction of $-g$, until reaching a terminal velocity where the size of each step is 
$$
\tag{8.17}
\frac{\epsilon ||g||}{1 - \alpha}.
$$
It is thus helpful to think of the momentum hyperparameter in terms of $\frac{1}{1 - \alpha}$. For example, $\alpha = 0.9$ corresponds to multiplying the maximum speed by 10 relative to the gradient descent algorithm.



## 8.3.3 Nesterov Momentum

The update rules in Nesterov momentum algorithm are given by
$$
\begin{eqnarray}
\tag{8.21}
v &\leftarrow& \alpha v - \epsilon \nabla_{\theta} \bigg(\frac{1}{m} \sum^m_{i=1} L(f(x^{(i)}; \theta + \alpha v), y^{(i)}) \bigg), \\

\tag{8.22}
\theta &\leftarrow& \theta + v,
\end{eqnarray}
$$

***

**Algorithm 8.3** Stochastic gradient descent (SGD) with Nesterov momentum

***

**Require:** Learning rate $\epsilon$, momentum parameter $\alpha$

**Require:** Initial parameter $\theta$, initial velocity $v$

​	**while** stopping criterion not met **do**

​		Sample a minibatch of $m$ examples from the training set {$x^{(1)}, \ldots, x^{(m)}$} with corresponding targets $y^{(i)}$

​		Apply interim update: $\tilde{\theta} \leftarrow \theta + \alpha v$.

​		Compute gradient (at interim point): $\hat{g} \leftarrow + \frac{1}{m} \nabla_{\tilde{\theta}} \sum_i L(f(x^{(i)}; \tilde{\theta}), y^{(i)})$.

​		Compute velocity update: $v \leftarrow \alpha v - \epsilon \hat{g}$.

​		Apply update: $\theta \leftarrow \theta + v$.

​	**end while**

***

where the parameters $\alpha$ and $\epsilon$ play a similar role as in the standard momentum method. The different between Nesterov momentum and standard momentum is where the gradient is evaluated. With Nesterov momentum, the gradient is evaluated after the current velocity is applied. Thus one can interpret Nesterov momentum as attempting to add a *correction factor* to the standard method of momentum.



# 8.5 Algorithms with Adaptive Learning Rates

The learning rate is reliably one of the most difficult to set hyperparameters because it significantly affects model performances. The cost is often highly sensitive to some directions in parameter space and insensitive to others. The momentum algorithm can mitigate these issues somewhat, but it does so at the expense of introducing another hyperparameter. Is there another way?

If we believe that the directions of sensitivity are somewhat axis aligned, it can make sense to use a separate learning rate for each parameter and automatically adapt these learning rate throughout the course of learning.

The **delta-bar-delta** algorithm(1988) is an early heuristic approach to adapting individual learning rates for model parameters during training. The approach is based on a simple idea: if the partial derivative of the loss, with respect to a given model parameter, remains the same sign, then the learning rate should increase. If that partial derivative changes sign, then the learning rate should decrease. Of course, this kind of rule can only be applied to full batch optimization.

More recently(2016), a number of incremental (or mini batch-based) methods have been introduced that adapt the learning rates of model parameters.

## 8.5.1 AdaGrad

***

**Algorithm 8.4** The AdaGrad algorithm

***

**Require:** Global learning rate $\epsilon$

**Require:** Initial parameter $\theta$

**Require:** Small constant $\delta$, perhaps $10^{-7}$, for numerical stability

​	Initialize gradient accumulation variable $r = 0$

​	**while** stopping criterion not met **do**

​		Sample a minibatch of $m$ examples from the training set {$x^{(1)}, \ldots, x^{(m)}$} with corresponding targets $y^{(i)}$.

​		Compute gradient estimate: $\hat{g} \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})$.

​		Accumulate squared gradient: $r \leftarrow r + \hat{g} \odot \hat{g}$

​		Compute update: $\Delta \theta \leftarrow -\frac{\epsilon}{\delta + \sqrt{r}} \odot \hat{g}$. (Division and square root applied element-wise)

​		Apply update: $\theta \leftarrow \theta + \Delta \theta$.

​	**end while**

***

The AdaGrad algorithm individually adapts the learning rates of all model parameters by **scaling them inversely proportional to the square root of the sum of all the historical squared values of the gradient**.

The parameters with largest partial derivative of the loss have a correspondingly rapid decrease in their learning rate, while parameters with small partial derivatives have a relatively small decrease in their learning rate. The net effect is greater progress in the more gently sloped directions of parameter space.

In the context of convex optimization, the AdaGrad algorithm enjoys some desirable theoretical properties. Empirically, however, for training deep neural network models, the accumulation of squared gradients *from the beginning of training* can result in a premature and excessive decrease in the effective learning rate. AdaGrad performs well for some but not all deep learning models.



## 8.5.2 RMSProp

The **RMSProp** algorithm(2012) modifies AdaGrad to perform better in the nonconvex setting by changing the gradient accumulation into an exponentially weighted moving average. AdaGrad is designed to converge rapidly when applied to a convex function. When applied to a nonconvex function to train a neural network, the learning trajectory may pass through many different structures and eventually arrive at a region that is a locally convex bowl. AdaGrad shrinks the learning rate according to the entire history of the squared gradient and may have made the learning rate too small before arriving at such a convex structure. RMSProp uses an exponentially decaying average to discard history from the extreme past so that it can converge rapidly after finding a convex bowl, as if it were an instance of the AdaGrad algorithm initialized within that bowl.

***

**Algorithm 8.5** The RMSProp algorithm

***

**Require:** Global learning rate $\epsilon$, decay rate $\rho$

**Require:** Initial parameter $\theta$

**Require:** Small constant $\delta$, perhaps $10^{-6}$, used to stabilize division by small numbers

​	Initialize gradient accumulation variable $r = 0$

​	**while** stopping criterion not met **do**

​		Sample a minibatch of $m$ examples from the training set {$x^{(1)}, \ldots, x^{(m)}$} with corresponding targets $y^{(i)}$.

​		Compute gradient estimate: $\hat{g} \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})$.

​		Accumulate squared gradient: $r \leftarrow \rho r + (1 - \rho)\hat{g} \odot \hat{g}$

​		Compute update: $\Delta \theta = -\frac{\epsilon}{\sqrt{\delta + r}} \odot \hat{g}$. ($\frac{1}{\sqrt{\delta + r}}$ applied element-wise)

​		Apply update: $\theta \leftarrow \theta + \Delta \theta$.

​	**end while**

***

***

**Algorithm 8.6** The RMSProp algorithm with Nesterov momentum

***

**Require:** Global learning rate $\epsilon$, decay rate $\rho$, momentum coefficient $\alpha$

**Require:** Initial parameter $\theta$, initial velocity $v$

**Require:** Small constant $\delta$, perhaps $10^{-6}$, used to stabilize division by small numbers

​	Initialize gradient accumulation variable $r = 0$

​	**while** stopping criterion not met **do**

​		Sample a minibatch of $m$ examples from the training set {$x^{(1)}, \ldots, x^{(m)}$} with corresponding targets $y^{(i)}$.

​		Compute interim update: $\tilde{\theta} \leftarrow \theta + \alpha v$.

​		Compute gradient (at interim point): $\hat{g} \leftarrow + \frac{1}{m} \nabla_{\tilde{\theta}} \sum_i L(f(x^{(i)}; \tilde{\theta}), y^{(i)})$.

​		Accumulate squared gradient: $r \leftarrow \rho r + (1 - \rho)\hat{g} \odot \hat{g}$

​		Compute velocity update: $v \leftarrow \alpha v -\frac{\epsilon}{\sqrt{r}} \odot \hat{g}$. ($\frac{1}{\sqrt{r}}$ applied element-wise)

​		Apply update: $\theta \leftarrow \theta + v$.

​	**end while**

***

Compared to AdaGrad, the use of the moving average introduces a new hyperparameter, $\rho$, that controls the length scale of the moving average.

Empirically, RMSProp has been shown to be an effective and practical optimization algorithm for deep neural networks ( A go-to optimization methods for deep learning practitioners).



## Adam

**Algorithm 8.7** The Adam algorithm

***

**Require:** Step size $\epsilon$ (Suggested default: 0.001)

**Require:** Exponential decay rates for moment estimate, $\rho_1$ and $rho_2$ in $[0, 1)$. (Suggested default: 0.9 and 0.999 respectively)

**Require:** Small constant $\delta$, used to stabilize division by small numbers (Suggested default: $10^{-8}$)

**Require:** Initial parameters $\theta$

​	Initialize 1st and 2nd moment variables $s = 0, r = 0$

​	Initialize time step $t = 0$

​	**while** stopping criterion not met **do**

​		Sample a minibatch of $m$ examples from the training set {$x^{(1)}, \ldots, x^{(m)}$} with corresponding targets $y^{(i)}$.

​		Compute gradient estimate: $\hat{g} \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})$.

​		$t \leftarrow t + 1$.

​		Update biased first moment estimate: $s \leftarrow \rho_1 s + (1 - \rho_1) \hat{g}$.

​		Update biased second moment estimate: $r \leftarrow \rho_2 r + (1 - \rho_2) \hat{g} \odot \hat{g}$.

​		Correct bias in first moment: $\hat{s} \leftarrow \frac{s}{1 - \rho_1^t}$.

​		Correct bias in first moment: $\hat{r} \leftarrow \frac{r}{1 - \rho_2^t}$.		

​		Compute update: $\Delta \theta = - \epsilon \frac{\hat{s}}{\delta + \sqrt{\hat{r}}}$. (operations applied element-wise)

​		Apply update: $\theta \leftarrow \theta + \Delta \theta$.

​	**end while**

***





