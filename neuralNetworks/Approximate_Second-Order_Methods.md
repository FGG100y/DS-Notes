# Approximate Second-Order Methods

For simplicity of exposition, the only objective function we examine is the empirical risk:
$$
\tag{8.25}
J(\theta) = \mathbb{E}_{\mathbf{x, y} \sim \hat{p}_{data}} [L(f(x; \theta), y)] = \frac{1}{m} \sum^m_{i=1} L(f(x^{(i)}; \theta), y^{(i)}).
$$
In contrast to first-order methods, second-order methods make use of the second derivatives to improve optimization. The most widely used second-order method is Newton's method.

## 8.6.1 Newton's Method

Newton's method is an optimization scheme based on using a second-order Taylor series expansion to approximate $J(\theta)$ near some point $\theta_0$, ignoring derivatives of higher order:
$$
\tag{8.26}
J(\theta) \approx J(\theta_0) + (\theta - \theta_0)^{\mathsf{T}} \nabla_{\theta} J(\theta_0) + \frac{1}{2} (\theta - \theta_0)^{\mathsf{T}} H (\theta - \theta_0),
$$
where $H$ is the Hessian of $J$ with respect to $\theta$ evaluated at $\theta_0$. If we then solve for the critical point of this function, we obtain the Newton parameter update rule:
$$
\tag{8.27}
\theta^* = \theta_0 - H^{-1} \nabla_{\theta} J(\theta_0).
$$
Thus for a locally quadratic function (with positive definite $H$), by rescaling the gradient by $H^{-1}$, Newton's method jumps directly to the minimum (NOTE that $\theta^*$ means the $\theta$ which optimizes the $J(\theta)$). If the objective function is convex but not quadratic (there are higher-order terms), this update can be iterated, yielding the training algorithm associated with Newton's method, given in algorithm 8.8.

For surfaces that are not quadratic, as long as the Hessian remains positive definite, Newton's method can be applied iteratively. This implies a two-step iterative procedure. First, update or compute the inverse Hessian (i.e., by updating the quadratic approximation). Second, update the parameters according to equation 8.27.

***

**Algorithm 8.8** Newton's method with objective $J(\theta) = \frac{1}{m} \sum^m_{i=1} L(f(x^{(i)}; \theta), y^{(i)})$.

***

**Require:** Initial parameter $\theta_0$

**Require:** Training set of m examples

​	**while** stopping criterion not met **do**

​		Compute gradient: $g \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})$.

​		Compute Hessian: $H \leftarrow \frac{1}{m} \nabla^2_{\theta} \sum_i L(f(x^{(i)}; \theta), y^{(i)})$.

​		Compute Hessian inverse: $H^{-1}$.

​		Compute update: $\Delta \theta = - H^{-1}g$.

​		Apply update: $\theta = \theta + \Delta \theta$.

​	**end while**

***

In previous section, we discuss how Newton's method is appropriate only when the Hessian is positive definite (No saddle point around). In deep learning, the surface of the objective function is typically nonconvex, with many features, such as saddle points, that are problematic for Newton's method. If the eigenvalues of the Hessian are not all positive, for example, near a saddle point, then Newton's method can actually cause updates to move in the wrong direction. This situation can be avoided by regularizing the Hessian. Common regularization strategies include adding a constant, $\alpha$, along the diagonal of the Hessian. The regularized update becomes
$$
\tag{8.28}
\theta^* = \theta_0 - [H(f(\theta_0)) + \alpha I]^{-1} \nabla_{\theta} J(\theta_0).
$$
Beyond the challenge created by certain features of the objective function, such as saddle points, the application of Newton's method for training large neural networks is limited by the significant computational burden it imposes. 



> 8.2.3
>
> For Newton's method, saddle points clearly constitute a problem. Gradient descent is designed to move "downhill" and is not explicitly designed to seek a critical point. Newton's method, however, is designed to solve for a point where the gradient is zero.
>
> The proliferation of saddle points in high-dimensional spaces presumably explains why second-order methods have not succeeded in replacing gradient descent for neural network training. Dauphin et al. (2014) introduced a **saddle-free Newton method** for second-order optimization and show that it improves significantly over the traditional version. Second-order methods remain difficult to scale to large neural networks, but this saddle-free approach holds promise if it can be scaled.



## BFGS, L-BFGS algorithms

