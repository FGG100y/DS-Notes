## 误差逆传播算法(error BackPropagation, BP)

BP 算法是迄今最成功的神经网络学习算法。现实任务中使用神经网络时，大多是在使用 BP 算法进行训练。BP 算法不仅可用于多层前馈神经网络(multi-layer feedforward neural networks)[^1] ，还可以用于其他类型的神经网络，例如训练递归神经网络。但通常说 “BP网络” 时，一般指用 BP 算法训练的多重前馈神经网络[^2]。

### BP网络及变量符号

给定训练集 $D = \{(x_1, y_1), \ldots, (x_m, y_m)\}, x_i \in \R^d, y_i \in \R^l$ ，即输入示例由 $d$ 个属性描述，输出 $l$ 维实值向量。为便于讨论，图5.7 给出了一个拥有 $d$ 个输入神经元、$l$ 个输出神经元、$q$ 个隐层神经元的多层前馈神经网络结构，其中

- 输出层第 $j$ 个神经元的阈值用 $\theta_j$ 表示，隐层第 $h$ 个神经元的阈值用 $\gamma_h$ 表示；

- 输入层第 $i$ 个神经元与隐层第 $h$ 个神经元之间的连接权为 $v_{ih}$ ；
- 隐层第 $h$ 个神经元与输出层第 $j$ 个神经元之间的连接权为 $w_hj$ ；
- 记隐层第 $h$ 个神经元接收到的输入为 $\alpha_h = \sum^d_{i=1} v_{ih} x_i$ ；
- 记输出层第 $j$ 个神经元接收到的输入为 $\beta_j = \sum^q_{h=1} w_{hj} b_h $ ，其中，$b_h$ 为隐层第 $h$ 个神经元的输出；
- 假设隐层和输出层神经元都使用sigmoid函数[^3] 作为激活函数[^4] 。

![BP network and notations](./images/xgs_BP_notations.png)

对训练例 $(x_k, y_k)$ ，假定神经网络的输出为 $\hat{y}_k = (\hat{y}^k_1, \ldots, \hat{y}^k_l)$ ，即
$$
\tag{5.3} \label{eq_y_hat}
\hat{y}^k_j = f(\beta_j - \theta_j),
$$
则网络在 $(x_k, y_k)$ 上均方误差为
$$
\tag{5.4}
E_k = {1 \over 2} \sum^l_{j=1} (\hat{y}^k_j - {y}^k_j)^2 .
$$
图5.7的网络中有 $(d + l + 1)q + l$ 个参数需确定：

- 输入层到隐层的 $d \times q$ 个权值；
- 隐层到输出层的 $q \times l$ 个权值;
- $q$ 个隐层神经元的阈值、$l$ 个输出层神经元的阈值。

BP 是一个迭代学习算法，在迭代的每一轮中采用广义的感知机学习规则[^5] 对参数进行更新估计，即任意参数 $v$ 的更新估计式为
$$
\tag{5.5}
v \leftarrow v + \Delta v .
$$

### 参数的更新

以下我们以图5.7中隐层到输出层的连接权 $w_{hj}$ 为例来进行推导。

BP 算法基于梯度下降(gradient descent)策略，以目标的的负梯度方向对参数进行调整。对式(5.4)的误差 $E_k$ ，给定学习率 $\eta$ ，有
$$
\tag{5.6}
\Delta w_{hj} = - \eta {\partial E_k \over \partial w_{hj}}.
$$
注意[^6]到 $w_{hj}$ 先影响到第 $j$ 个输出层神经元的输入值 $\beta_j$ ，再影响到其输出值 $\hat{y}^k_j$ ，然后影响到 $E_k$ ，有
$$
\tag{5.7}
\frac{\partial E_k}{\partial w_{hj}} =
\frac{\partial E_k}{\partial \hat{y}^k_j}
\cdot \frac{\partial \hat{y}^k_j}{\partial \beta_j}
\cdot \frac{\partial \beta_j}{\partial w_{hj}} .
$$
根据 $\beta_j$ 的[定义](###BP网络及变量符号)，显然有
$$
\tag{5.8}
\frac{\partial \beta_j}{\partial w_{hj}} = b_h .
$$
sigmoid函数有一个很好的性质：
$$
\tag{5.9}
f'(x) = f(x)(1 - f(x)),
$$
于是，根据式(5.4)和式(5.3)，有
$$
\begin{eqnarray}
g_j
&=& - \frac{\partial E_k}{\partial \hat{y}^k_j} \cdot \frac{\partial \hat{y}^k_j}{\partial \beta_j} \\
&=& - (\hat{y}^k_j - {y}^k_j) f'(\beta_j - \theta_j) \\
\tag{5.10} \label{eq_output_gradient}
&=& \hat{y}^k_j(1 - \hat{y}^k_j) (\hat{y}^k_j - {y}^k_j) .
\end{eqnarray}
$$
将式(5.10)和(5.8)代入式(5.7)，再代入式(5.6)，就得到了 BP 算法中关于 $w_{hj}$ 的更新公式
$$
\tag{5.11} \label{eq_weights_hj}
\Delta w_{hj} = \eta g_j b_h .
$$
类似可得
$$
\begin{eqnarray}
\tag{5.12}
\Delta \theta_j &=& - \eta g_j, \\
\tag{5.13}
\Delta v_{ih} &=& \eta e_h x_i, \\
\tag{5.14} \label{eq_threshold_hidden}
\Delta \gamma_h &=& - \eta e_h, \\
\end{eqnarray}
$$
其中
$$
\begin{eqnarray}
e_h
&=& - \frac{\partial E_k}{\partial b_h} \cdot \frac{\partial b_h}{\partial \alpha_h} \\
&=& - \sum^l_{j=1} \frac{\partial E_k}{\partial \beta_j} \cdot \frac{\partial \beta_j}{\partial b_h} f'(\alpha_h - \gamma_h) \\
&=& \sum^l_{j=1} w_{hj} g_j f'(\alpha_h - \gamma_h) \\
\tag{5.15} \label{eq_hidden_gradient}
&=& b_h (1 - b_h) \sum^l_{j=1} w_{hj} g_j .
\end{eqnarray}
$$
学习率 $\eta \in (0, 1)$ 控制着算法每一轮迭代中的更新步长，若太大则容易振荡，太小则收敛速度又会过慢。

有时为了做精细调节，可令式(5.11)与(5.12)使用 $\eta_1$ ，式(5.13)与(5.14)使用 $\eta_2$ ，两者未必相等。

### BP 算法的工作流程

对每个训练样例，BP 算法执行以下操作：先将输入示例提供给输入层神经元，然后逐层将信号前传，直到产生输出层的结果；然后计算输出层的误差(第4-5行)，再将误差逆向传播至隐层神经元(第6行)，最后根据隐层神经元的误差来对连接权和阈值进行调整(第7行)。该迭代过程循环进行，直到达到某些停止条件为止。

---

`BP 算法工作流程` 

---

**输入**：训练集 $D = \{(x_k, y_k)\}^m_{k=1}$ ；

​			学习率 $\eta$ .

**过程**：

1：在 $(0,1)$ 范围内随机初始化网络中所有连接权和阈值

2：**repeat**

3: 		**for all** $(x_k, y_k) \in D$ **do**

4: 				根据当前参数和式($\ref{eq_y_hat}$)计算当前样本的输出 $\hat{y}_k$ ；

5: 				根据式($\ref{eq_output_gradient}$)计算输出层神经元的梯度项 $g_j$ ；

6: 				根据式($\ref{eq_hidden_gradient}$)计算隐层神经元的梯度项 $e_h$ ；

7: 				根据式($\ref{eq_weights_hj}$)-($\ref{eq_threshold_hidden}$)更新连接权 $w_{hj}, v_{ih}$ 与阈值 $\theta_j, \gamma_h$ 

8: 		**end for** 

9: **unitl** 达到停止条件

**输出**：连接权与阈值确定的多层前馈神经网络

---

**需要注意的是**，BP 算法的目标是要最小化训练集 $D$ 上的累积误差
$$
\tag{5.16}
E = {1 \over m} \sum^m_{k=1} E_k ,
$$
但我们上面介绍的 “标准BP算法” 每次仅针对一个训练样例更新连接权和阈值，也就是说 `BP 算法工作流程` 中算法的更新规则是基于单个的 $E_k$ 推导而得。如果类似地推导出基于累积误差最小化的更新规则，就得到了 “累积误差逆传播(accumulated error backpropagation)” 算法。累积BP算法与标准BP算法都很常用。一般来说，标准BP算法每次更新只针对单个样例，参数更新得非常频繁，而且对不同样例进行更新的效果可能出现 “抵消” 现象。因此，为了达到同样的累积误差极小点，标准BP算法往往需要进行更多次数的迭代。累积BP算法直接针对累积误差最小化，它在读取整个训练集 $D$ 一遍后才对参数进行更新，其参数更新频率低得多。但在很多任务中，累积误差下降到一定程度之后，进一步下降会非常缓慢，这时标准BP算法往往会更快获得较好的解，尤其是在训练集 $D$ 非常大时更为明显[^7] [^8]。











[^1]: 神经网络层级结构的一种，每层神经元与下一层神经元全互连，神经元之间不存在同层连接，也不存在跨层连接。“前馈”并不意味着网络中信号不能向后传，而是指网络拓扑结构上不存在环或回路。

[^2]: The term back-propagation (BP) is often misunderstood as meaning the whole learning algorithm for multi layer neural networks. Actually, back-propagation refers only to the method for computing the gradient, while another algorithm, such SGD, is used to perform learning using this gradient.
[^3]: $\text{sigmoid}(x) = {1 \over 1 + e^{-x}}$ ，对数几率函数是典型的sigmoid函数，它把可能在较大范围内变化的输入值挤压到 $(0,1)$ 输出值范围内，因此有时也称为 “挤压函数(squashing function)”。
[^4]: 激活函数也称为 “响应函数”，理想的激活函数是阶跃函数 $\text{sgn}(x)_{x |x < 0} = 0; \text{sgn}(x)_{x |x \ge 0} =1$，它将输入值映射为输出值 "0" 或 “1”，显然 “1” 对应于神经元兴奋，“0” 对应于神经元兴奋。然而，阶跃函数具有不连续(在零点处)、不光滑等不太友好的性质，因此实际常用sigmoid函数作为激活函数。
[^5]: 感知机(Perceptron)由两层神经元组成，输入层接收外界输入信号后传递给输出层，输出层是 M-P 神经元，也称为 “阈值逻辑单元(threshold logic unit)”。感知机的学习规则非常简单，对训练样例 $(x, y)$ ，若当前感知机的输出为 $\hat{y}$ ，则感知机权重将这样调整：$w_i \leftarrow w_i + \Delta w_i$ ，其中 $\Delta w_i = \eta (y - \hat{y}) x_i$ ，其中 $\eta \in (0,1)$ 是学习率(learning rate)。可以看出，若感知机对训练样例 $(x, y)$ 预测正确，即 $\hat{y} = y$ ，则感知机不发生变化，否则将根据错误的程度进行权值调整。
[^6]: 这就是 “链式法则”。
[^7]: 标准BP算法和累积BP算法的区别类似于随机梯度下降(SGD)与标准梯度下降之间的区别。
[^8]: 感知机的参数更新规则和BP算法的参数更新规则式(5.11)-(5.14) 都是基于梯度下降。