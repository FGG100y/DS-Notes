

```html
注明：
框架内容均来自周志华的西瓜书，真正的大师之作。
其他添补内容均单独注释引用源。
```

## 回归模型 (Regression Model)

### 基本形式

给定由 $d$ 个属性描述的示例 ${x} = (x_1, \ldots, x_d)$ ，其中，$x_i$ 是 $x$ 在第 $i$ 个属性上的取值，线性模型(linear model)试图学得一个通过属性的线性组合来进行预测的函数，一般用向量形式写成
$$
\tag{3.2} \label{eq:lm}
f(x) = w^{\sf{T}}x + b,
$$
其中，$w = (w_1, \ldots, w_d)$ 。$w$ 和 $b$ 学得之后，模型就得以确定。

### 线性回归

给定数据集 $D = \{(x_1, y_1), \ldots, (x_m, y_m) \}$ ，其中 $x_i = (x_{i1}, \ldots, x_{id}), y_i \in \R$ 。“线性回归(linear regression)” 试图学得一个线性模型以尽可能准确地预测实值输出标记。

#### 简单线性回归

**只有一个输入属性时**，数据集 $D = \{(x_i, y_i) \}^m_{i=1}$ ，其中，$x_i \in \R$ 。对离散属性，若属性值间存在 “序(order)” 关系，可通过连续化将其转化为连续值，例如二值属性 “身高” 的取值 “高” “矮” 可转化为 $\{1.0, 0.0 \}$ ；三值属性 “高度 = ｛高，中，低｝”  转化为 “高度 = $\{1.0, 0.5, 0.0 \}$” ；若属性值间不存在序关系，假定存在 $k$ 个属性值，则通常将其转化为 $k$ 维向量，例如属性 “瓜类 = ｛西瓜，南瓜，黄瓜｝” 转化为 “瓜类 = ｛$(0,0,1), (0,1,0), (1,0,0)$｝”。

简单线性回归试图学得
$$
\tag{3.3}
f(x_i) = wx_i + b,\ \text{使得} f(x_i) \simeq y_i.
$$
如何确定 $w$ 和 $b$ 呢？显然，关键在于如何衡量 $f(x)$ 与 $y$ 之间的差别。均方误差是回归任务中最常用的性能度量，因此我们试图让均方误差最小化，即
$$
\begin{eqnarray}
(w^*, b^*)
&=& argmin_{(w,b)} \sum^m_{i=1} (f(x_i) - y_i)^2 \\
\tag{3.4}
&=& argmin_{(w,b)} \sum^m_{i=1} (y_i - wx_i - b)^2 .
\end{eqnarray}
$$
均方误差有非常好的几何意义，它对应了常用的欧几里得距离(即 “欧氏距离(Euclidean distance)”)。基于均方误差最小化来进行模型求解的方法称为 “最小二乘法(least square method)[^1]” 。在线性回归中(从坐标系角度来看)，最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧氏距离之和最小。



> 在预测任务中，要评估学习器 $f$ 的性能，就要把学习器预测结果 $f(x)$ 与真实标记 $y$ 进行比较。均方误差(mean square error)也叫作 “平方损失(square loss)” ，
> $$
> \tag{2.2}
> E(f;D) = {1 \over m} \sum^m_{i=1} (f(x_i) - y_i)^2
> $$
> 更一般的，对于数据分布 $\cal{D}$ 和概率密度函数 $p(·)$ ，均方误差可描述为
> $$
> \tag{2.3}
> E(f;D) = \int_{x \in \cal{D}} (f(x) - y)^2 p(x)dx.
> $$



求解 $w$ 和 $b$ 使得 $E_{(w,b)} = \sum^m_{i=1} (y_i - wx_i - b)^2$ 最小化的过程，称为线性回归模型的最小二乘 “参数估计(parameter estimation)” 。我们可将 $E_{(w,b)}$ 分别对 $w$ 和 $b$ 求导，得到
$$
\begin{eqnarray}
\tag{3.5}
\frac{\partial E_{(w,b)}}{\partial w}
&=& 2 \bigg(w \sum^m_{i=1} x_i^2 - \sum^m_{i=1} (y_i - b) x_i \bigg), \\
\\
\tag{3.6}
\frac{\partial E_{(w,b)}}{\partial b}
&=& 2 \bigg(mb - \sum^m_{i=1} (y_i - wx_i) \bigg),
\end{eqnarray}
$$
然后令式3.5和式3.6分别为零[^2]，可得到 $w$ 和 $b$ 最优解的闭式(closed-form)解
$$
\begin{eqnarray}
\tag{3.7}
w &=& \frac{\sum^m_{i=1} y_i(x_i - \bar{x})}{\sum^m_{i=1}x_i^2 - {1 \over m}(\sum^m_{i=1}x_i)^2}  \\
\\
\tag{3.8}
b &=& {1 \over m} \sum^m_{i=1} (y_i - wx_i)
\end{eqnarray}
$$
其中，$\bar{x} = {1 \over m} \sum^m_{i=1}x_i$ 为 $x$ 的均值。

> The *closed-form* : What and How? 
>
> All the details following are quoted from [The Method of Least Square](https://web.williams.edu/Mathematics/sjmiller/public_html/BrownClasses/54/handouts/MethodLeastSquares.pdf) .
>
> Differentiating $E_{(w,b)}$ yields
> $$
> \begin{eqnarray}
> 
> \frac{\partial E_{(w,b)}}{\partial w}
> &=& \sum^m_{i=1} 2 (y_i - (wx_i + b)) \cdot (-x_i), \\
> \\
> \tag{a3.12}
> \frac{\partial E_{(w,b)}}{\partial b}
> &=& \sum^m_{i=1} 2 (y_i - (wx_i + b)) \cdot 1.
> \end{eqnarray}
> $$
> Setting the partial derivatives to $0$ (and dividing by 2) yields
> $$
> \begin{eqnarray}
> 
> \sum^m_{i=1} (y_i - (wx_i + b)) \cdot x_i
> &=& 0, \\
> \\
> \tag{a3.13}
> \sum^m_{i=1} (y_i - (wx_i + b))
> &=& 0.
> \end{eqnarray}
> $$
> We may rewrite these equations as
> $$
> \begin{eqnarray}
> 
> \bigg(\sum^m_{i=1} x_i^2 \bigg)w + \bigg(\sum^m_{i=1} x_i \bigg)b
> &=& \sum^m_{i=1} x_i y_i, \\
> \\
> \tag{a3.14}
> \bigg(\sum^m_{i=1} x_i \bigg)w + \bigg(\sum^m_{i=1} 1 \bigg)b
> &=& \sum^m_{i=1} y_i.
> \end{eqnarray}
> $$
> Turns these to the following matrix equation:
> $$
> \tag{a3.15}
> \begin{pmatrix}
> 	\sum^m_{i=1}x_i^2 	&\sum^m_{i=1}x_i \\
> 	\sum^m_{i=1}x_i 	&\sum^m_{i=1}1 
> \end{pmatrix}
> \begin{pmatrix}
> 	w \\
> 	b 
> \end{pmatrix}
> =
> \begin{pmatrix}
> 	\sum^m_{i=1}x_i y_i \\
> 	\sum^m_{i=1}y_i 
> \end{pmatrix}
> $$
> We will show the matrix is invertible, which implies
> $$
> \tag{a3.16}
> 
> \begin{pmatrix}
> 	w \\
> 	b 
> \end{pmatrix}
> =
> \begin{pmatrix}
> 	\sum^m_{i=1}x_i^2 	&\sum^m_{i=1}x_i \\
> 	\sum^m_{i=1}x_i 	&\sum^m_{i=1}1 
> \end{pmatrix}^{-1}
> \begin{pmatrix}
> 	\sum^m_{i=1}x_i y_i \\
> 	\sum^m_{i=1}y_i 
> \end{pmatrix}
> $$
> Denote the matrix by $M$. The determinant of $M$ is
> $$
> \tag{a3.17}
> 
> \det{M} = 
> {\sum^m_{i=1} x_i^2 \cdot \sum^m_{i=1} 1} - {\sum^m_{i=1} x_i \cdot \sum^m_{i=1} x_i}.
> $$
> As we know that
> $$
> \tag{a3.18}
> \bar{x} = {1 \over m} \sum^m_{i=1} x_i,
> $$
> we find that
> $$
> \tag{a3.19}
> \begin{eqnarray}
> 	\det{M}
> 	&=& m \sum^m_{i=1} x_i^2 - (m \bar{x})^2 \\
> 	&=& m^2 \bigg({1 \over m}\sum^m_{i=1} x_i^2 - \bar{x}^2 \bigg) \\
> 	&=& m^2 \cdot {1 \over m}\sum^m_{i=1}(x_i - \bar{x})^2,
> \end{eqnarray}
> $$
> where the last equality follows from simple algebra. Thus, as long as all the $x_i$ are not equal, $\det{M}$ will be non-zero and $M$  will be invertible.
>
> ---
>
> **Thus we find that, so long as the $x$'s are not all equal, the best fit values of $w$ and $b$ are obtained by solving a linear system of equations; the solution is given in (3.16).**
>
> ---
>
> The quote end.

#### 多元线性回归

更一般的情形是如本节开头的数据集 $D$ ，样本由 $d$ 个属性描述。此时我们试图学得
$$
f(x_i) = w^{\sf{T}} x_i + b,\ \text{使得}\ f(x_i) \simeq y_i,
$$
这称为 “多元线性回归(multivariate linear regression)”。

类似的，可用最小二乘法来对 $w$ 和 $b$ 进行估计。为便于讨论，我们把 $w$ 和 $b$ 吸收如向量形式 $\mathbf{w}= (w,b)$ ，相应的，把数据集 $D$ 表示为一个 $m \times (d + 1)$  大小的矩阵 $\bf{X}$ ，其中每行对应一个示例，该行前 $d$ 个元素对应于示例的 $d$ 个属性值，最后一个元素恒置为 $1$[^3] ，即
$$
\bf{X}
= \left( \begin{array}{ccc}
x_{11} & x_{12} & \ldots & x_{1d} & 1 \\
x_{21} & x_{22} & \ldots & x_{2d} & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
x_{m1} & x_{m2}& \cdots  & x_{md} & 1
\end{array} \right)
=
\left( \begin{array}{ccc}
x_1^{\sf{T}} & 1 \\
x_2^{\sf{T}} & 1 \\
\vdots & \vdots \\
x_m^{\sf{T}} & 1
\end{array} \right),
$$
再把标记也写成向量形式
$$
\bf{y}
= \left( 
\begin{array}{ccc}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{array}
\right),
$$
则类似于式3.4，有
$$
\tag{3.9}
\bf{w}^* = argmin_{(w)} (\bf{y} - Xw)^{\sf{T}} (\bf{y} - Xw) .
$$
令 $E_{\bf{w}} = (\bf{y} - Xw)^{\sf{T}} (\bf{y} - Xw)$ ，对 $\bf{w}$ 求导得到
$$
\tag{3.10}
\frac{\partial E_{\bf{w}}}{\partial \bf{w}}
= 2 \bf{X}^{\sf{T}}(\bf{Xw} - y).
$$
令式3.10为零可得 $\bf{w}$ 最优解的闭式解，但由于涉及矩阵逆的计算，比单变量情形要复杂一些。以下进行简单讨论

当 $\bf{X}^{\sf{T}}X$ 为满秩矩阵(full-rank matrix)[^4]或正定矩阵(positive definite matrix)[^5]时，令式3.10为零可得
$$
\tag{3.11}
\bf{w}^* = (\bf{X}^{\sf{T}}X)^{-1} \bf{X}^{\sf{T}}y ,
$$
其中 $(\bf{X}^{\sf{T}}X)^{-1}$ 是 $\bf{X}^{\sf{T}}X$ 的逆矩阵。令 $\bf{x} = (x_i, 1)$ ，则最终学得的多元线性回归模型为
$$
\tag{3.12}
f(\mathbf{x}_i) = \mathbf{x}_i^{\sf{T}} (\bf{X}^{\sf{T}}X)^{-1} \bf{X}^{\sf{T}}y .
$$
然而，现实任务中 $\bf{X}^{\sf{T}}X$ 往往不是满秩矩阵。例如在生物信息的基因芯片数据中常有成千上万个属性，但往往只有几十上百个样例，这会导致 $\bf{X}$ 的列数多于行数， $\bf{X}^{\sf{T}}X$ 显然不满秩。此时可解出多个 $\bf{w}$ ，它们都能使均方误差最小化。选择哪一个解作为输出，将由学习算法的归纳偏好决定，常见的做法是引入正则化项(regularization term)。





[^1]: 最小二乘法(Ordinary Least Square) 用途很广，不仅限于线性回归。
[^2]: 函数在导数为零处取得最小值或最大值。
[^3]: $\bf{X}$ 最后一列恒置为 $1$ ，因为$ \mathbf{w}= (w,b)$ ，从而保证截距项($b$)在矩阵乘法中保持一致。
[^4]: A full rank matrix is one which has linearly independent rows or/and linearly independent columns. if you were to find the RREF(Row Reduced Echelon Form) of a full rank matrix, then it would contain all 1s in its main diagonal -- that is all the pivot positions are occupied by 1s only.
[^5]: A positive definite matrix is one which all its eigenvalues are positive.



#### 广义线性模型

线性模型虽然简单，却有丰富的变化。例如对于样例 $(x, y),\  y \in \R$ ，当我们希望线性模型($\ref{eq:lm}$)的预测值逼近真实标记 $y$ 时，就得到了线性回归模型。

可否令模型预测值逼近 $y$ 的衍生物呢？比如说，假设我们认为示例所对应的输出标记是在指数尺度上变化，那就可将输出标记的对数作为线性模型逼近的目标，即
$$
\tag{3.14}
\text{ln} y = w^{\sf{T}}x + b.
$$
这就是 “对数线性回归(log-linear regression)”，它实际上是试图让 $e^{w^{\sf{T}}x + b}$ 逼近 $y$ 。式3.14在形式上仍是线性回归，但实质上已是在求取输入空间到输出空间的非线性函数映射，如图3.1所示。这里的对数函数起到了将线性回归模型的预测值与真实标记联系起来的作用。

![llr](./figs/ml_regression_model.png)

更一般地，考虑单调可微函数 $g(·)$ [^6]，令
$$
\tag{3.15}
y = g^{-1}(w^{\sf{T}}x + b),
$$
这样得到的模型称为 “广义线性模型(generalized linear model)”，其中函数 $g(·)$ 称为 “联系函数(link function)”。显然，对数线性回归是广义线性模型在 $g(·) = \text{ln}(·)$ 时的特例。广义线性模型的参数估计通常通过加权最小二乘法或极大似然法进行。

[^6]: “单调可微”也就是要求$g(·)$ 连续且充分光滑。 



### 对数几率回归

前面主要讨论了如何使用线性模型进行回归学习，但若要做的是**分类任务**该怎么办？答案蕴含在式3.15的广义线性模型中：只需要找一个单调可微函数将分类任务的真实标记 $y$ 与线性回归模型的预测值联系起来。

考虑二分类任务，其输出标记 $y \in \{0, 1 \}$ ，而线性回归模型产生的预测值 $z = w^{\sf{T}}x + b$ 是实值，于是，我们需要将实值 $z$ 转换为 $0/1$ 值。最理想的是 “单位阶跃函数(unit-step function)”
$$
\tag{3.16}
y = \left\{ 
\begin{array}{ll}
0, & z < 0;& \\
0.5, & z = 0;& \\
1, & z > 0;& \\
\end{array} \right.
$$
即若预测值 $z$ 大于零就判为正例，小于零则判为负例，预测值为临界值则可任意判别，如图3.2所示。

![lor](./figs/ml_regression_model_4_classification.png)

从图3.2可看出，单位阶跃函数不连续(在零点处不连续)，因此不能直接用作式3.15中的 $g^{-1}(·)$ 。于是，我们希望找到能在一定程度上近似单位阶跃函数的 “替代函数(surrogate function)” ，并希望它满足单调可微。对数几率函数(简称 “**对率函数**”[^7]，logistic function)正是这样一个常用的替代函数：
$$
\tag{3.17}
y = \frac{1}{1 + e^{-z}}.
$$
从图3.2可看出，对数几率函数是一种 “Sigmoid函数”(即形似S的函数)，它将 $z$ 值转化为一个接近 $0$ 或 $1$ 的 $y$ 值，并且其输出值在 $z = 0$ 附近变化很陡。将对数几率函数作为 $g^{-1}(·)$ 代入式3.15，得到
$$
\tag{3.18}
y = \frac{1}{1 + e^{-(w^{\sf{T}}x + b)}},
$$
类似于式3.14，式3.18可变化为
$$
\tag{3.19}
\text{ln}{y \over 1-y} = w^{\sf{T}}x + b
$$
若将 $y$ 视为样本 $x$ 作为正例的可能性，则 $1 - y$ 是其反例的可能性，两者的比值
$$
\tag{3.20}
{y \over 1-y}
$$
被称为 “几率(odds)”，反映了 $x$ 作为正例的相对可能性。对几率取对数则得到 “对数几率(log odds, a.k.a., logit)”
$$
\tag{3.21}
\text{ln}{y \over 1-y}.
$$
由此可看出，式3.18实际上是在用线性回归模型的预测值去逼近真实标记的对数几率，因此其对应的模型称为 “对数几率回归(logistic regression, a.k.a., logit regression)”[^8]。

需要注意的是，虽然它的名字带有 “回归”，但实际却是一种分类学习方法。这种方法有很多优点，例如它是直接对**分类可能性**进行建模，无需事先假设数据分布，这样就避免了假设不准确带来的问题；它不是仅仅预测出 “类别” ，而是可得到近似概率预测，这对许多利用概率辅助决策的任务很有用；此外，对率函数是任意阶可导[^9]的凸函数[^10]，有很好的数学性质，现有的许多数值优化算法都可直接用于求取最优解。



[^7]: 对数几率函数(对率函数)与 “对数函数” $\text{ln}(·)$ 不同。
[^8]: 有文献翻译为 “逻辑回归”，但中文 “逻辑” 与 logistic/logit 的含义--对数几率--相去甚远。
[^9]:  ${d \over dx} e^x = e^x$ ，因此，任意阶可导。
[^10]: 对区间 $[a, b]$ 上定义的函数 $f$ ，若它对区间中任意两点 $x_1, x_2$ 均有 $f({x_1 + x_2 \over 2}) \leq {f(x_1) + f(x_2) \over 2}$ ，则称 $f$ 为区间$[a, b]$ 上的凸函数。对实数集上的函数，可通过求二阶导数来判断：若二阶导数在区间上非负，则为凸函数；若二阶导数在区间上恒大于0，则为严格凸函数。



下面我们来看看如何确定式3.18中的 $w$ 和 $b$ 。

若将式3.18中的 $y$ 视为“类后验概率估计” $p(y = 1 | x)$ ，则式3.19可重写为
$$
\tag{3.22}
\text{ln}{p(y = 1 |x) \over p(y = 0 |x)} = w^{\sf{T}}x + b
$$
显然有（也就是可以构造出）[^11][^14][^15]
$$
\begin{eqnarray}
\tag{3.23}
p(y = 1 |x) &=& \frac{e^{w^{\sf{T}}x + b}}{1 + e^{w^{\sf{T}}x + b}}, \\
\\
\tag{3.24}
p(y = 1 |x) &=& \frac{1}{1 + e^{w^{\sf{T}}x + b}}. \\
\end{eqnarray}
$$
于是，我们可以通过 “极大似然法(MLE)” 来估计 $w$ 和 $b$ 。数据集 $D = \{(x_i, y_i) \}^m_{i=1}$ ，对率回归模型最大化 “对数似然(log-likelihood)”
$$
\tag{3.25}
\mathcal{L}(w,b) = \sum^m_{i=1} \text{ln}\ p(y_i | x_i;w,b)
$$

即令每个样本属于其真实标记的概率越大越好。为便于讨论，令 $\beta = (w,b), \mathbf{x} = (x, 1)$ ，则 $w^{\sf{T}}x + b$ 可简写为 $\beta^{\sf{T}} \mathbf{x}$ 。再令 $p_1(\mathbf{x}, \beta) = p(y = 1 | \mathbf{x}, \beta)$ ， $p_0(\mathbf{x}, \beta) = p(y = 0 | \mathbf{x}, \beta ) = 1- p_1(\mathbf{x}, \beta)$ ，则式3.25中的似然项可重写为[^12][^13]
$$
\tag{3.26}
p(y_i | x_i;w,b) = y_i p_1(\mathbf{x}, \beta) + (1 - y_i) p_0(\mathbf{x}, \beta).
$$
将式3.26代入式3.25，并根据式3.23和式3.24可知，最大化式3.25等价于最小化
$$
\tag{3.27}
\mathcal{L}(\beta)= \sum^m_{i=1} \bigg(
-y_i \beta^{\sf{T}} \mathbf{x}_i + \text{ln}(1 + e^{\beta^{\sf{T}} \mathbf{x}_i})
\bigg)
$$
式3.27是关于 $\beta$ 的高阶可导连续凸函数，根据凸优化理论，经典的梯度下降法、牛顿法等都可以求得其最优解，于是就得到
$$
\tag{3.28}
\beta^* = argmin_{(\beta)} \cal{L}(\beta)
$$





[^11]: 将式3.23比上式3.24再取自然对数，即可得到式3.22. 可是，用来构造这个式子的因子是胡乱猜测得来的吗？或者是暴力枚举？

[^12]: (伯努利分布中) $y_i$ 要么为 $0$ ，要么为 $1$ 。
[^13]: 二项分布(binomial distribution)用以描述 $N$ 次独立的伯努利实验中有 $m$ 次成功(即$x = 1$)的概率，其中每次伯努利实验成功的概率为 $\mu \in [0, 1]$ 。当 $N = 1$ 时，二项分布退化为伯努利分布。

[^14]: To avoid this problem(that using simple linear regression to predict the probability of binary classes or multiple classes would result in values beyond [0, 1]), we must model $p(X)$ using a function that gives outputs between $0$ and $1$ for all values of $X$. Many functions meet this description. In logistic regression, we use the *logistic function* : $p(X)=\frac{\exp(\beta_0 + \beta_1 X)}{1 + \exp(\beta_0 + \beta_1 X)}$. From book 'An Introduction to Statistical learning', p.132.
[^15]: And where does this *logistic function* come from, how come the equation form? It rooted in the *chaos theory* with another form of $[A] x_{t+1} = k x_t \cdot (1-x_t)$, where $x_t$ is the population of a certain species at generation $t$; while $x_{t+1}$ is the population of a certain species at the next generation. In Wikipedia it is $[B] f(x) = {L \over {1 + \exp(-k(x-x_0))}}$. That equation $[B]$ comes from a **differential** version of $[A]$, here is how: Subtract $x_t$ to the LHS and RHS of $[A]$ to get: $x_{t+1} - x_t = k x_t(1-x_t) - k{1 \over k}x_t \Rightarrow \frac{x_{t+1} - x_t}{\Delta{t}}=k'x_t(1 - Lx_t)\ \text{with}\ L :=1+{1 \over k}$. Which is analogous to $[A]$, and one being discrete, the other continuous. From [here](https://math.stackexchange.com/questions/3328730/logistic-function-where-does-it-come-from). 