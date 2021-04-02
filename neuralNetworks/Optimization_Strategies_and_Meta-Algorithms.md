# Optimization Strategies and Meta-Algorithms

Many optimization techniques are not exactly algorithms but rather general templates that can be specialized to yield algorithms, or subroutines that can be incorporated into many different algorithms.

## 8.7.1 Batch Normalization

BN is actually not an optimization algorithm, it is a method of adaptive reparametrization, motivated by the difficultly of training very deep models.

Very deep models involve the composition of several functions, or layers. The gradient tells how to update each parameter, under the assumption that the other layers do not change. In practice, we update all the layers simultaneously[^1].

[^1]: We typically train a neural network with gradient descent and back-propagation. Gradient descent to update parameters iteratively and backprop to compute the gradient (which is used by the gradient descent!) of the loss function with respect to each of these parameters. Let $w = [w_1, \ldots, w_n] \in \R^n$ be the vector that contains all learnable parameters of the network, $J(w)$ be the loss function, $\epsilon$ be the learning rate. Recall that the gradient descent step to update parameters is $w \leftarrow w - \epsilon \nabla_w J(w)$, we are assigning to $w$ the value $w - \epsilon \nabla_w J(w)$, so we are updating all parameters $w$ simultaneously, so we are also updating all layers simultaneously.**[Think in vectors, matrices, not scalars]**

