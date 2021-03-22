[toc]



# Back-Propagation Algorithms

When we use a feedforward neural network to accept an input $x$ and produce an output $\hat{y}$, information flows forward through the network. The input $x$ provides the initial information that then propagates up to the hidden units at each layer and finally produces $\hat{y}$. This is called **forward propagation**. During training, forward propagation can continue onward until it produces a scalar cost $J(\theta)$. The **back-propagation** algorithm, often simply called **backprop**, allows the information from the cost to then flow backward through the network in order to compute the gradient.

Computing an analytical expression for the gradient is straightforward, but numerically evaluating such an expression can be computationally expensive. The back-propagation algorithm does so using a simple and inexpensive procedure.

The term back-propagation is often misunderstood as meaning the whole learning algorithm for multi layer neural networks. **Actually, back-propagation refers only to the method for computing the gradient, while another algorithm, such SGD, is used to perform learning using this gradient.**

Furthermore, back-propagation is often misunderstood as being specific to multilayer neural networks, but in principle it can compute derivatives of any function ( for some function, the correct response is to report that the derivative of the function is undefined).

Specifically, we will describe how to compute the gradient $\nabla_{x} f(x, y)$ for an arbitrary function $f$, where $x$ is a set of variables whose derivatives are desired, and $y$ is an additional set of variables that are inputs to the function but whose derivatives are not required. In learning algorithms, the gradient we most often require is the gradient of the cost function with respect to the parameters, $\nabla_{\theta} J(\theta)$.

## 6.5.1 Computational Graphs

To describe the back-propagation algorithm more precisely, it is helpful to have a more precise **computational graph** language.

Here we use each node in the graph to indicate a variable. The variable may be a scalar, vector, matrix, tensor, or even a variable of another type. We also need to introduce the idea of an **operation**. An operation is a simple function of one or more variables. Our graph language is accompanied by a set of allowable operations. Functions more complicated than the operations in this set may be described by composing many operations together.

Without loss of generality, we define an operation to return only a single output variable (Note that the output can be a vector). If a variable $y$ is computed by applying an operation to a variable $x$, then we draw a directed edge from $x$ to $y$ (with annotation if needed).  Example of computational graph are shown in Figure 1.

![Computational Graph](./images/dl_computational_graphs.png)

Figure 1: (a) $z = x y$. (b) $\hat{y} = \sigma (x^{\mathsf{T}} w + b)$, some of the intermediate expressions do not have names in algebraic expression but need names in the graph would be simply name the $i$-th such variable $u^{(i)}$. (c) $H = \text{max} \{0, XW + b \}$. (d) Here we show a computation graph that applies more than one operation to the weights $w$ of a linear regression model. The weights are used to make both the prediction $\hat{y}$ and the weight decay penalty $\lambda \sum_i w_i^2$.



## 6.5.2 Chain Rule of Calculus

The chain rule of calculus is used to compute the derivatives of functions formed by composing other functions whose derivatives are known. **Back-propagation is an algorithm that computes the chain rule, with a specific order of operations that is highly efficient.**

Let $x$ be a real number, and let $f$ and $g$ both be functions mapping from a real number to a real number. Suppose that $y= g(x)$ and $z = f(g(x)) = f(y)$. Then the chain rule states that
$$
\tag{6.44}
\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}.
$$
We can generalize this beyond the scalar case. Suppose that $x \in \R^m, y \in \R^n$, $g$ maps from $\R^m$ to $\R^n$, and $f$ maps from $\R^n$ to $\R$. if $y= g(x)$ and $z = f(y)$, then
$$
\tag{6.45}
\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}
$$
In vector notation, this may be equivalently written as
$$
\tag{6.46}
\nabla_x z = \bigg(\frac{\partial y}{\partial x} \bigg)^{\mathsf{T}} \nabla_y z,
$$
where $\frac{\partial y}{\partial x}$ is the $n \times m$ Jacobian matrix of $g$.

**From this we see that the gradient of a variable $x$ can be obtained by multiplying a Jacobian matrix $\frac{\partial y}{\partial x}$ by a gradient $\nabla_y z$. The back-propagation algorithm consists of performing such a Jacobian-gradient product for each operation in the graph.**

Usually we apply the back-propagation algorithm to tensors of arbitrary dimensionality, not merely to vectors. Conceptually, this is exactly the same as back-propagation with vectors. The only difference is how the numbers are arranged in a grid to form a tensor. We could imagine flattening each tensor into a vector before we run back-propagation, computing a vector-valued gradient, and then reshaping the gradient back into a tensor. In this rearranged view, back-propagation is still just multiplying Jacobians by gradients.

To denote the gradient of a value $z$ with respect to a tensor $\mathsf{X}$, we write $\nabla_{\mathsf{X}} z$, just as if  $\mathsf{X}$ were a vector. The indices into  $\mathsf{X}$ now have multiple coordinates, for example, a 3-D tensor is indexed by three coordinates. We can abstract this away by using a single variable $i$ to represent the complete tuple of indices. For all possible index tuples $i$, $(\nabla_{\mathsf{X}}z)_i$ gives $\frac{\partial z}{\partial \mathsf{X}_i}$. Using this notation, we can write the chain rule as it applies to tensors. If $\sf{Y} = g(\sf{X})$ and $z = f(\sf{Y})$, then
$$
\tag{6.47}
\label{eq_ChainRule}
\nabla_{\sf{X}} z = \sum_j (\nabla_{\sf{X}} \sf{Y}_j) \frac{\partial z}{\partial \sf{Y}_j}.
$$

## 6.5.3 Recursively Applying the Chain Rule to Obtain Backprop

We begin with a version of the back-propagation algorithm that specifies the actual gradient computation directly (algorithm 6.2 along with algorithm 6.1 for the associated forward computation), in the order it will actually be done and according to the recursive application of chain rule.

First consider a computational graph describing how to compute a single scalar $u^{(n)}$ (say, the loss on a training example). This scalar is the quantity whose gradient we want to obtain, with the respect to the $n_i$ input nodes $u^{(1)}$ to $u^{(n_i)}$. In other words, we wish to compute $\frac{\partial u^{(n)}}{\partial u^{(i)}}$ for all $i \in \{1, 2, \ldots, n_i \}$. In the application of back-propagation to computing gradient for gradient descent over parameters, $u^{(n)}$ will be the cost associated with an example or a minibatch, while  $u^{(1)}$ to $u^{(n_i)}$ correspond to the parameters of the model.

We will assume that the nodes of the graph have been ordered in such a way that we can compute their output one after the other, starting at $u^{(n_i + 1)}$ and going up to  $u^{(n)}$. As define in algorithm 6.1, each node  $u^{(i)}$ is associated with an operation  $f^{(i)}$ and is computed by evaluating the function
$$
\tag{6.48}
u^{(i)} = f(\mathbb{A}^{(i)})
$$
where $\mathbb{A}^{(i)}$ is the set of all nodes that are parents of $u^{(i)}$.

**That algorithm specifies the forward propagation computation, which we could put in a graph $\cal{G}$. To perform back-propagation, we construct  a computational graph that depends on $\cal{G}$ and adds to it an extra set of nodes. These form a subgraph $\cal{B}$ with one node per node of $\cal{G}$. Computation in $\cal{B}$ proceeds in exactly the reverse of the order of computation in $\cal{G}$, and each node of $\cal{B}$ computes the derivative $\frac{\partial u^{(n)}}{\partial u^{(i)}}$ associated with the forward graph node $u^{(i)}$. This is done using the chain rule with respect to scalar output $u^{(n)}$:
$$
\tag{6.49}
\frac{\partial u^{(n)}}{\partial u^{(j)}} = \sum_{i:j \in Pa(u^{(i)})} \frac{\partial u^{(n)}}{\partial u^{(i)}} \frac{\partial u^{(i)}}{\partial u^{(j)}}
$$
as specified by algorithm 6.2. The subgraph $\cal{B}$ contains exactly one edge for each edge from node $u^{(j)}$ to node $u^{(i)}$ of $\cal{G}$. The edge from  $u^{(j)}$ to $u^{(i)}$ is associated with the computation of $\frac{\partial u^{(i)}}{\partial u^{(j)}}$. In addition, a dot product is preformed for each node, between the gradient already computed with respect to node $u^{(i)}$ that are children of  $u^{(j)}$ and the vector containing the partial derivatives $\frac{\partial u^{(i)}}{\partial u^{(j)}}$ for the same children nodes $u^{(i)}$.

The back-propagation algorithm is designed to reduce the number of common subexpressions without regard to memory. Specifically, it performs on the order of on Jacobian product per node in the graph. This can be seen from the fact that backprop (algorithm 6.2) visits each edge from node  $u^{(j)}$ to node $u^{(i)}$ of the graph exactly once in order to obtain the associated partial derivative $\frac{\partial u^{(i)}}{\partial u^{(j)}}$. Back-propagation thus avoids the exponential explosion in repeated subexpressions.

***

**Algorithm 6.1** A procedure that performs the computations mapping $n_i$ inputs  $u^{(1)}$ to $u^{(n_i)}$ to an output $u^{(n)}$. This defines a computational graph where each node computes numerical value $u^{(i)}$ by applying a function $f^{(i)}$ to the set of arguments $\mathbb{A}^{(i)}$ that comprises the values of previous nodes $u^{(j)}, j < i$, with $j \in Pa(u^{(i)})$. The input to the computational graph is the vector $x$, and is set into the first $n_i$ nodes  $u^{(1)}$ to $u^{(n_i)}$. The output of the computational graph is read off the last (output) node $u^{(n)}$.

***

**for** $i = 1, \ldots, n_i$ **do**

​	$u^{(i)} \leftarrow x_i$

**end for**

**for** $i = n_i + 1, \ldots, n$ **do**

​	$\mathbb{A}^{(i)} \leftarrow \{u^{(j)} | j \in Pa(u^{(i)}) \}$

​	$u^{(i)} \leftarrow f^{(i)}(\mathbb{A}^{(i)})$

**end for**

**return** $u^{(n)}$

***

***

**Algorithm 6.2** Simplified version of the back-propagation algorithm for computing the derivatives of $u^{(n)}$ with respect to the variables in the graph. Each $\frac{\partial u^{(i)}}{\partial u^{(j)}}$ is a function of the parents $u^{(j)}$ of $u^{(i)}$, thus linking the nodes of the forward graph to those added for the back-propagation graph.

***

Run forward propagation (algorithm 6.1 for this example) to obtain the activations of the network.

Initialize $\mathsf{\text{grad_table}}$, a data structure that will store the derivatives that have been computed. The entry $\mathsf{\text{grad_table}}[u^{(i)}]$ will store the computed value of $\frac{\partial u^{(n)}}{\partial u^{(i)}}$.

$\mathsf{\text{grad_table}}[u^{(i)}] \leftarrow 1$

**for** $j = n - 1$ down to $1$ **do**

​	The next line computes $\frac{\partial u^{(n)}}{\partial u^{(j)}} = \sum_{i:j \in Pa(u^{(i)})} \frac{\partial u^{(n)}}{\partial u^{(i)}} \frac{\partial u^{(i)}}{\partial u^{(j)}}$ using stored values:

​	$\mathsf{\text{grad_table}}[u^{(i)}] \leftarrow \sum_{i:j \in Pa(u^{(i)})} \mathsf{\text{grad_table}}[u^{(i)}] \frac{\partial u^{(i)}}{\partial u^{(j)}}$

**end for**

**return** $\{ \mathsf{\text{grad_table}}[u^{(i)}] | i = 1, \ldots, n_i \}$

***



>  A computational graph that results in repeated subexpressions when computing the gradient. Let $w \in \R$ be the input to the graph. We use the same function $f: \R \rightarrow \R$ as the operation that we apply at every step of a chain: $x = f(w), y = f(x), z = f(y)$. To compute $\frac{\partial z}{\partial w}$, we apply equation 6.44 and obtain:
> $$
> \begin{eqnarray}
> \tag{6.50}
> &&\frac{\partial z}{\partial w} \\
> \tag{6.51}
> &=& \frac{\partial z}{\partial y}\frac{\partial y}{\partial x}\frac{\partial x}{\partial w} \\
> \tag{6.52}
> &=& f'(y)f'(x)f'(w) \\
> \tag{6.53}
> &=& f'(f(f(w)))f'(f(w))f'(w).
> \end{eqnarray}
> $$
> Equation 6.52 suggests an implementation in which we compute the value of $f(w)$ only once and store it in the variable $x$. This is the approach taken by the back-propagation algorithm. (Equation 6.53 is also a valid implementation of the chain rule and is useful when memory is limited)
>
> ![backprop](./images/back-propagation.png)
>
> Figure 6.10: An example of the symbol-to-symbol approach to computing derivatives. In this approach, the back-propagation algorithm does not need to ever access any actual specific numeric values. Instead, it adds nodes to a computational graph describing how to compute these derivatives. A generic graph evaluation engine can later compute the derivatives for any specific numeric values. (Left) In this example, we begin with a graph representing $z = f(f(f(w)))$. (Right) We run the back-propagation algorithm, instructing it to construct the graph for the expression corresponding to $\frac{\partial z}{\partial w}$. In this example, we do not explain how the back-propagation algorithm works. The purpose is only to illustrate what the desired result is: a computational graph with a symbolic description of the derivative.

## 6.5.6 General Back-Propagation

The back-propagation algorithm is very simple. To compute the gradient of some scalar $z$ with respect to one of its ancestors $x$ in the graph, we begin by observing that the gradient with respect to $z$ is given by $\frac{dz}{dz} = 1$. We can then compute the gradient with respect to each parent of $z$ in the graph by multiplying the current gradient by the Jacobian[^1][^2] of the operation that produced $z$. We continue multiplying by Jacobians, traveling backward through the graph in this way until we reach $x$. For any node that may be reached by going backward from $z$ through two or more paths, we simply sum the gradients arriving from different paths at that node.

More formally, each node in the graph $\cal{G}$ corresponds to a variable. To achieve maximum generality, we describe this variable as being a tensor $\sf{V}$. Tensors in general can have any number of dimensions. They subsume scalars, vectors, and matrices.

We assume that each variable $\sf{V}$ is associated with the following subroutines:

- $\text{get_operation}(\sf{V})$: This returns the operation that computes $\sf{V}$, represented by the edges coming into  $\sf{V}$ in the computational graph. For example, there may be a Python or C++ class representing the matrix multiplication operation, and the $\text{get_operation}$ function. Suppose we have a variable that is created by matrix multiplication, $C = AB$. The $\text{get_operation}(\sf{V})$ returns a pointer to an instance of the corresponding C++ class.
- $\text{get_consumers}(\sf{V}, \cal{G})$: This returns the list of variables that are children of $\sf{V}$ in the computational graph $\cal{G}$.
- $\text{get_inputs}(\sf{V}, \cal{G})$: This returns the list of variables that are parents of $\sf{V}$ in the computational graph $\cal{G}$.

Each operation $\text{op}$ is also associated with a $\text{bprop}$ operation. This $\text{bprop}$ operation can compute a Jacobian-vector product as described by equation $\ref{eq_ChainRule}$. This is how the back-propagation algorithm is able to achieve great generality. Each operation is responsible for knowing how to back-propagate through the edges in the graph that it participates in. For example, we might use a matrix multiplication operation to create a variable $C = AB$. Suppose that the gradient of a scalar $z$ with respect to $C$ is given by $G$. The matrix multiplication operation is responsible for defining two back-propagation rules, one for each of its input arguments. If we call the $\text{bprop}$ method to request the gradient with respect to $A$ given that the gradient on the output is $G$, then the $\text{bprop}$ method of the matrix multiplication operation must state that the gradient with respect to $A$ is given by $GB^{\sf{T}}$. Likewise, $A^{\sf{T}}G$ the the gradient with respect to $B$. 

### backprop rule

The back-propagation algorithm itself does not need to know any differentiation rules. It only needs to call each operation's $\text{bprop}$ rules with the right arguments. Formally, $\text{op.bprop(inputs}, \sf{X}, \sf{G})$ must return
$$
\tag{6.54}
\sum_i (\nabla_{\sf{X}} \text{op.f(inputs)}_i) \sf{G}_i,
$$
which is just an implementation of the chain rule as expressed in equation $\ref{eq_ChainRule}$. Here, $\text{inputs}$ is a list of inputs that are supplied to the operation, $\text{op.f}$ is the mathematical function that the operation implements, $\sf{X}$ is the input whose gradient we wish to compute, and $\sf{G}$ is the gradient on the output of the operation.



***

**Algorithm 6.5** The outermost skeleton of the back-propagation algorithm. This portion does simple setup and cleanup work. Most of the important work happens in the $\text{build_grad}$ subroutine of algorithm 6.6.

***

**Require:** $\mathbb{T}$, the target set of variables whose gradients must be computed.

**Require:** $\cal{G}$, the computational graph

**Require:** $z$, the variable to be differentiated

​	Let $\cal{G}'$ be $\cal{G}$ pruned to contain only nodes that are ancestors of $z$ and descendants of nodes in $\mathbb{T}$.

​	Initialize $\sf{\text{grad_table}}$, a data structure associating tensors to their gradients $\mathsf{\text{grad_table}}[z] \leftarrow 1$.

​	**for** $\sf{V}$ in $\mathbb{T}$ **do**

​		$\sf{\text{build_grad}}(\sf{V}, \cal{G}, \cal{G}', \sf{\text{grad_table}})$

​	**end for**

​	**Return** $\sf{\text{grad_table}}$ restricted to $\mathbb{T}$

***

***

**Algorithm 6.6** The inner loop subroutine $\sf{\text{build_grad}}(\sf{V}, \cal{G}, \cal{G}', \sf{\text{grad_table}})$ of the back-propagation algorithm, called by the back-propagation algorithm defined in algorithm 6.5.

***

**Require:** $\sf{V}$, the variable whose gradient should be added to $\cal{G}$ and $\sf{\text{grad_table}}$.

**Require:** $\cal{G}$, the computational graph to modify

**Require:** $\cal{G}'$, the restriction of $\cal{G}$ to nodes that participate in the gradient

**Require:** $\sf{\text{grad_table}}$, a data structure mapping nodes to their gradients

​	**if** $\sf{V}$ is in $\sf{\text{grad_table}}$ **then**

​		**Return** $\sf{\text{grad_table}}[\sf{V}]$

​	**end if**

​	$i \leftarrow 1$

​	**for** $\sf{C}$ in $\text{get_consumers}(\sf{V}, \cal{G}')$ **do**

​		$\text{op} \leftarrow \text{get_operation}(\sf{C})$

​		$\sf{D} \leftarrow \sf{\text{build_grad}}(\sf{C}, \cal{G}, \cal{G}', \sf{\text{grad_table}})$

​		$\sf{G}^{(i)} \leftarrow \text{op.bprop}(\text{get_inputs} (\sf{C}, \cal{G}'), \sf{V}, \sf{D})$

​		$i \leftarrow i + 1$

​	**end for**

​	$\sf{G} \leftarrow \sum_i \sf{G}^{(i)}$

​	$\sf{\text{grad_table}}[\sf{V}] = \sf{G}$

​	Insert $\sf{G}$ and the operations creating it into $\cal{G}$

​	**Return** $\sf{G}$

***



[^1]: Sometimes we need to find all the partial derivatives of a function whose input and output are both vectors. The matrix containing all such partial derivatives is known as a **Jacobian matrix**. Specifically, if we have a function: $f: \R^m \rightarrow \R^n$, then  the Jacobian matrix $J \in \R^{n \times m}$ of $f$ is defined such that $J_{i,j} = \frac{\partial}{\partial x_j} f(x)_i$.
[^2]: For functions with multiple inputs, we must make use of the concept of **partial derivatives**. The partial derivative $\frac{\partial}{\partial x_i} f(x)$ measures how $f$ changes as only the variable $x_i$ increases at point $x$. The **gradient** generalizes the notion of derivative to the case where the derivative is with respect to a vector: the gradient of $f$ is the vector containing all the  partial derivatives, denoted $\nabla_x f(x)$. Element $i$ of the gradient is the partial derivative of $f$ with respect to $x_i$.



## 6.5.7 Example: Back-Propagation for MLP Training

Here we develop a very simple MLP with a single hidden layer. To train this model, we will use minibatch SGD. The back-propagation algorithm is used to compute the gradient of the cost on a single minibatch. Specifically, we use a minibatch of examples from the training set formatted as a design matrix $X$ and a vector of associated class labels $y$. The network computes a layer of hidden features $H = \text{max}(0, XW^{(1)})$. To simplify the presentation we do not use biases in this model. We assume that our graph language includes a $\text{relu}$ operation that can compute max{0, $Z$} element-wise. The predictions of the unnormalized log probabilities over classes are then given by $HW^{(2)}$. We assume that our graph language includes a $\text{cross-entropy}$ operation that computes the cross-entropy between the targets $y$ and the probability distribution defined by these unnormalized log probabilities. The resulting cross-entropy defines the cost $J_{MLE}$. Minimizing this cross-entropy performs maximum likelihood estimation of the classifier. However, to make this example more realistic, we also include a regularization term. The total cost
$$
\tag{6.56}
J = J_{MLE} + \lambda \bigg( {\sum_{i,j} \big(W^{(1)}_{i,j} \big)^2} + {\sum_{i,j} \big(W^{(2)}_{i,j} \big)^2} \bigg)
$$
consists of the cross-entropy and a weight decay term with coefficient $\lambda$. The computational graph is illustrated in figure 6.11.

![single-layer-MLP-training](./images/dl_mlp_training_graph.png)

Figure 6.11: The computational graph used to compute the cost to train our example of a single-layer MLP using the cross-entropy loss and weight decay. (NOTE that the graph of gradient of this example is large enough that it would be tedious to draw or to read.)

We can roughly trace out the behavior of the back-propagation algorithm by looking at the forward propagation graph in figure 6.11. To train, we wish to compute both $\nabla_{W^{(1)}}J$ and $\nabla_{W^{(2)}}J$. There are two different paths leading backward from $J$ to weights: one through the cross-entropy cost, and one through the weight decay cost. 

- **The path through weight decay cost:** this is relatively simple, it will always contribute $2 \lambda W^{(i)}$ to the gradient on $W^{(i)}$.
- **The path through cross-entropy cost:** this is slightly more complicated. Let $G$ be the gradient on the unnormalized log probabilities $U^{(2)}$ provided by the $\text{cross-entropy}$ operation. The back-propagation algorithm now needs to explore two different branches. On the shorter branch, it adds $H^{\sf{T}}G$ to the gradient on $W^{(2)}$, using the [back-propagation rule](###backprop-rule) for the second argument to the matrix multiplication operation. The other branch corresponds to the longer chain descending further along the network. First, the back-propagation algorithm computes $\nabla_H J = G W^{(2) \sf{T}}$ using the [back-propagation rule](###backprop-rule) for the first argument to the matrix multiplication operation. Next, the $\text{relu}$ operation uses its  [back-propagation rule](###backprop-rule) to zero out components of the gradient corresponding to entries of $U^{(1)}$ that are less than $0$. Let the result be called $G'$. The last step of the back-propagation algorithm is to use the [back-propagation rule](###backprop-rule) for the second argument of the $\text{matmul}$ operation to add $X^{\sf{T}} G'$ to the gradient on $W^{(1)}$.

**After these gradients have been computed, the SGD or another optimization algorithm, uses these gradients to update the parameters.**

For the MLP, the computational cost is dominated by the cost of matrix multiplication. 

- During the forward propagation stage, we multiply by each weight matrix, resulting in $O(w)$ multiply-adds, where $w$ is the number of weights.
- During the backward propagation stage, we multiply by the transpose of each weight matrix, which has the same computational cost.

The main memory cost of the algorithm is that we need to store the input to the nonlinearity of the hidden layer. This value is stored from the time it is computed until the backward pass has returned to the same point. The memory cost is thus $O(m n_h)$, where $m$ is the number of examples in the minibatch and $n_h$ is the number of hidden units.





