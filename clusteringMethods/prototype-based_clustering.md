[toc]



## 原型聚类

原型[^1]聚类也称为 “基于原型的聚类(prototype-based clustering)”，此类算法假设聚类结构能够通过一组原型刻画，在现实聚类任务中极为常用。通常情形下，算法先对原型进行初始化，然后对原型进行迭代更新求解。采用不同的原型表示、不同的求解方式，将产生不同的算法。

### $k$ 均值聚类算法

给定样本集 $D = \{x_1, \ldots, x_m \}$ ，“$k$ 均值($k$-means)” 算法针对聚类所得簇划分 $\mathcal{C} = \{C_1, \ldots, C_k \}$ 最小化平方误差
$$
\tag{9.24}
E = \sum^k_{i=1} \sum_{x \in C_i} ||x - \mu_i||^2_2 ,
$$
其中，$\mu_i = {1 \over |C_i|} \sum_{x \in C_i} x$ 是簇 $C_i$ 的均值向量。直观来看，式(9.24)在一定程度上刻画了簇内样本围绕簇均值向量的紧密程度，$E$ 值越小则簇内样本相似度越高。

最小化式(9.24)并不容易，找到它的最优解需考察样本集 $D$ 的所有可能簇划分，这是一个 NP 难问题[^2] 。因此，$k$ 均值算法采用了贪心策略，通过迭代优化来近似求解式(9.24)。算法流程如下，其中第1行对均值向量进行初始化，在第4-8行与第9-16行依次对当前簇划分及均值向量迭代更新，若迭代更新后聚类结果保持不变，则在第18行将当前的簇划分结果返回。

---

`k 均值算法流程` 

---

**输入**：样本集 $D = \{x_1, \ldots, x_m \}$ ；

​			聚类簇数 $k$ 

**过程**：

1: 从 $D$ 中随机选择 $k$ 个样本作为初始均值向量 $\{\mu_1, \ldots, \mu_k \}$ 

2: **repeat**

3: 		令 $C_i = \phi \quad (1 \le i \le k)$ 

4: 		**for** $j = 1, 2, \ldots, m$ **do**

5: 				计算样本 $x_j$ 与各个均值向量 $\mu_i \ (1 \le i \le k)$ 的距离：$d_{ji} = ||x_j - \mu_i||_2$ ；

6: 				根据距离最近的均值向量确定 $x_j$ 的簇标记[^3]：$\lambda_j = \text{argmin}_{i \in \{1, 2, \ldots, k\}} d_{ji}$ ；

7: 				将样本 $x_j$ 划入相应的簇：$C_{\lambda_j} = C_{\lambda_j} \cup \{x_j\}$ ；

8: 		**end for** 

9: 		**for** $i = 1, 2, \ldots, k$ **do** 

10: 				计算新均值向量：$\mu_i' = {1 \over |C_i|} \sum_{x \in C_i} x$ ；

11: 				**if** $\mu_i' \ne \mu_i$ **then** 

12: 						将当前均值向量 $\mu_i$ 更新为 $\mu_i'$ 

13: 				**else** 

14: 						保持当前均值向量不变

15: 				**end if**

16: 	 **end for** 

17: **until** 当前均值向量均未更新

**输出**：簇划分 $\mathcal{C} = \{C_1, C_2, \ldots, C_k \}$ 

---



[^1]: “原型” 是指样本空间中具有代表性的点。
[^2]: NP hardness problem ([non-deterministic polynomial-time](https://wikimili.com/en/NP_(complexity)) hardness): wait
[^3]: $\lambda_j$ 实际是 $\{1, 2, \ldots, k \}$ 中的某个数，是 $k$ 个不同聚类簇 $\cal{C}$ 的下标，因为 $\text{argmin}_{i \in \{1, 2, \ldots, k\}} (\text{expression}_i)$ 函数就是返回使得 $\text{expression}$ 最小的那个 $i$ 。



### K-Means in Action

#### Finding the optimal number of clusters

In general, it will not be easy to know how to set $k$, and the result might be quite bad if you set it to the wrong value (see Figure 9-7).

![cluster No.](./images/handson_cluster_numbers.png)

You might be thinking that we could just pick the model with lowest inertia. Unfortunately, the inertia is not a good performance metric when trying to choose $k$ because it keeps getting lower as we increase $k$. Indeed, the more clusters there are, the closer each instance will be to its closest centroid, and therefore the lower the inertia will be (see Figure 9-8: a plot of the inertia as a function of $k$).

![cluster inertia](./images/handson_cluster_inertia.png)

This technique for choosing the best value for the number of clusters is rather coarse. A more precise approach (but more computationally expensive) is to use the **silhouette score**, which is the mean silhouette coefficient (轮廓系数) over all the instances. An instance's silhouette coefficient is equal to $(b - a) / \max(a, b)$, where $a$ is the mean distance to the other instances in the same cluster (i.e., the mean intra-cluster distance) and $b$ is the mean nearest-cluster distance (i.e., the mean distance to the instances of the next cluster, defined as the one that minimizes $b$, excluding the instance's own cluster).

The silhouette coefficient can vary between -1 and +1.

- silhouette coefficient close to +1 means that the instance is well inside its own cluster and far from other clusters
- silhouette coefficient close to 0 means that it is close to a cluster boundary
- silhouette coefficient close to -1 means that the instance may have been assigned to the wrong cluster

To compute the silhouette score, you can use sklearn's `silhouette_score()` function:

```python
from sklearn.metrics import silhouette_score

# give it all the instances in the dataset
# and the labels they were assigned
silhouette_score(X, kmeans_model.labels_)
```

We can also compare the silhouette scores for different numbers of clusters (see Figure 9-9):

![silhouette scores](./images/handson_cluster_silhouette_scores.png)

As the Figure 9-9 shows, this visualization is much richer than the previous one: although it confirms that $k = 4$ is a very good choice, it also underlines the fact that $k = 5$ is quite good as well, and much better than $k > 5$. This is not visible when comparing the inertias.

**silhouette diagram** 

An even more informative visualization is obtained when you plot every instance's silhouette coefficient, sorted by the cluster they are assigned to and by the value of the coefficient. This is called a *silhouette diagram* (see Figure 9-10). Each diagram contains one knife shape per cluster. The shape's height indicates the number of instances the cluster contains, and its width represents the sorted silhouette coefficients of the instances in the cluster (wider is better). The dashed line indicated the mean silhouette coefficient.

![silhouette coefficient sorted](./images/handson_cluster_silhouette_coefficients_sorted.png)

The dashed line represents the mean silhouette score for each number of clusters. When most of the instances in a cluster have a lower coefficient than this score, then the cluster is rather bad since this means its instances are much too close to other clusters (such as when $k = 3, k = 6$).  But when $k =4$ or $k = 5$ ,  the clusters look pretty good: most instances extend beyond the dashed line. When $k = 4$ , the cluster at index 1 is rather big. When $k = 5$ , all clusters have similar sizes. So, even though the overall silhouette score from $k=4$ is slightly greater than for $k=5$ , it seems like a good idea to use $k=5$ to get clusters of similar sizes.

#### Limits of K-Means

Despite its many merits, most notably being fast and scalable, K-Means is not perfect. As we saw, it is necessary to run the algorithm several times to avoid suboptimal solutions, plus you need to specify the number of clusters, which can be quite a hassle. Moreover, K-Means does not behave very well when the clusters have varying sizes, different densities, or non-spherical shapes (see Figure 9-11).

![cluster shapes](./images/handson_cluster_shapes.png)

As Figure 9-11 shows, neither of these solutions is any good (the solution on the right is just terrible even though its inertia is lower). So, depending on the data, different clustering algorithms may perform better. On these types of elliptical clusters, *Gaussian mixture models* work great.



### 学习向量量化

与 $k$ 均值算法类似，“学习向量量化(Learning Vector Quantization, LVQ)” 也是试图找到一组原型向量来刻画聚类结构，但与一般聚类算法不同的是，LVQ 假设数据样本带有类别标记[^4][^5] ，学习过程利用样本的这些监督信息来辅助聚类。

给定样本集 $D = \{(x_1, y_1), \ldots, (x_m, y_m) \}$ ，每个样本 $x_j$ 是由 $n$ 个属性描述的特征向量 $(x_{j1}; x_{j2}; \ldots;x_{jn})$ ， $y_j \in \mathcal{Y}$ 是样本 $x_j$ 的类别标记。LVQ 的目标是学得一组 $n$ 维原型向量 $\{p_1, p_2, \ldots, p_q \}$ ，每个原型向量代表一个聚类簇，簇标记 $t_i \in \mathcal{Y}$ 。

`LVQ 算法` 描述如下：算法第1行先对原型向量进行初始化，例如对第q个簇可以从类别标记为 $t_q$ 的样本中随机选取一个作为原型向量。算法第2-12行对原型向量进行迭代优化。在每一轮迭代中，算法随机选取一个有标记的训练样本，找出与其距离最近的原型向量，并根据两者的类别标记是否一致来对原型向量进行相应的更新。在第12行中，若算法的停止条件已满足（例如达到最大迭代轮数，或原型向量趋于稳定），则将当前原型向量作为最终结果返回。

---

`LVQ 算法`

---

**输入**：样本集 $D = \{(x_1, y_1), \ldots, (x_m, y_m) \}$ ；

​			原型向量个数 $q$ ，各原型向量预设类别标记 $\{ t_1, t_2, \ldots, t_q\}$ ；

​			学习率 $\eta \in (0, 1)$ .

**过程**：

01: 初始化一组原型向量 $\{p_1, p_2, \ldots, p_q \}$ 

02: **repeat**

03: 		从样本集 $D$ 随机选取样本 $(x_j, y_j)$ ；

04: 		计算样本 $x_j$ 与 $p_i (1 \le i \le q)$ 的距离：$d_{ji} = ||x_j - p_i||_2$ ；

05: 		找出与 $x_j$ 距离最近的原型向量 $p_{i*}$ ，$i* = \text{argmin}_{i \in \{1, 2, \ldots, q\}} d_{ji}$ ；

06: 		**if** $y_j = t_{i*}$ **then**

07: 				$p' = p_{i*} + \eta \cdot (x_j - p_{i*})$ 

08: 		**else**

09: 				$p' = p_{i*} - \eta \cdot (x_j - p_{i*})$ 

10: 		**end if** 

11: 		将原型向量 $p_{i*}$ 更新为 $p'$ 

12: **until** 满足停止条件

**输出**：原型向量 $\{p_1, p_2, \ldots, p_q \}$ 

---

显然，LVQ 的关键是第6-10行，即如何更新原型向量。直观上看，对样本 $x_j$ ，若最近的原型向量 $p_{i*}$ 与 $x_j$ 的类别标记相同，则令 $p_{i*}$ 与 $x_j$ 的方向靠拢，如`LVQ 算法` 第7行所示，此时新原型向量为
$$
\tag{9.25}
p' = p_{i*} + \eta \cdot (x_j - p_{i*}) ,
$$
$p'$ 与 $x_j$ 之间的距离为
$$
\begin{eqnarray}
||p' - x_j||_2
&=& ||p_{i*} + \eta \cdot (x_j - p_{i*}) - x_j||_2 \\
\\
\tag{9.26}
&=& (1 - \eta) \cdot ||p_{i*} - x_j||_2 .
\end{eqnarray}
$$
令学习率 $\eta \in (0, 1)$ ，则原型向量 $p_{i*}$ 在更新为 $p'$ 之后将更接近 $x_j$ 。

类似的，若 $p_{i*}$ 与 $x_j$ 的类别标记不同，则更新后的原型向量与 $x_j$ 之间的距离将增大 $(1 + \eta) \cdot ||p_{i*} - x_j||_2$ ，从而更远离 $x_j$ 。

在学得一组原型向量 $\{p_1, p_2, \ldots, p_q \}$ 后，即可实现对样本空间 $\cal{X}$ 的簇划分。对任意样本 $x$ ，它将被划入与其距离最近的原型向量所代表的簇中；换言之，每个原型向量 $p_i$ 定义了一个与之相关的一个区域 $R_i$[^6] ，该区域中每个样本与 $p_i$ 的距离不大于它与其他原型向量 $p_{i'} (i \ne i')$ 的距离，即
$$
\tag{9.27}
R_i = \{x \in \mathcal{X} \quad \text{so that} \quad ||x - p_i||_2 \le ||x - p_{i'}||_2, i' \ne i \} .
$$
由此形成了对样本空间 $\cal{X}$ 的簇划分 $\{R_1, R_2, \ldots, R_q \}$ ，该划分通常称为 “Voronoi 剖分(Voronoi tessellation)” 。



[^4]: SOM 是基于无标记样本的聚类算法，而 LVQ 可看作 SOM 基于监督信息的扩展。SOM（Self-Organizing Map, 自组织映射）网络是一种竞争型学习(competitive learning) 的无监督神经网络，它能将高维输入数据映射到低维空间(通常二维) ，同时保持输入数据在高维空间的拓扑结构，即将高维空间中相似的样本点映射到网络输出层的邻近神经元。
[^5]: 竞争型学习是神经网络中常用的一种无监督学习策略，在使用该策略时，网络的输出神经元互相竞争，每一时刻仅有一个竞争获胜的神经元被激活，其他神经元的状态被抑制（“胜者通吃(winner-take-all)原则”）。
[^6]: 若将 $R_i$ 中样本全用原型向量 $p_i$ 表示，则可实现数据的 “有损压缩(lossy compression)”，这称为 “向量量化( vector quantization)” ；LVQ 由此而得名。



## 高斯混合聚类(GMM)

与 $k$ 均值、LVQ 用原型向量类刻画聚类结构不同，高斯混合(Mixture-of-Gaussian) 聚类算法采用概率模型来表达聚类原型。

> 简单回顾 | 多元高斯分布
>
> 多元高斯分布的定义：对 $n$ 维样本空间 $\cal{X}$ 中随机向量 $x$ ，若 $x$ 服从高斯分布，其概率密度函数为
> $$
> \tag{9.28}
> p(x) = {1 \over {(2 \pi)^{n \over 2} |\Sigma|^{1 \over 2}}} \exp {\bigg(-{1\over2}(x - \mu)^{\mathsf{T}} \Sigma^{-1} (x - \mu) \bigg)} ,
> $$
> 其中，$\exp(x) = e^x$ ，$\mu$ 是 $n$ 维均值向量，$\Sigma$ 是 $n \times n$ 的协方差矩阵(并且是“对称正定矩阵”， 正定矩阵意思是其eigenvalues都大于零)，$|\Sigma|$ 是其行列式，$\Sigma^{-1}$ 是其逆矩阵。由式(9.28)可看出，高斯分布完全由均值向量 $\mu$ 和协方差矩阵 $\Sigma$ 这两个参数确定。为了明确显示高斯分布与相应参数的依赖关系，将概率密度函数记为 $p(x | \mu, \Sigma)$ 。

我们可以定义高斯混合分布[^7]
$$
\tag{9.29}
p_{\cal{M}} (x) = \sum^k_{i=1} \alpha_i \cdot p(x | \mu_i, \Sigma_i) ,
$$
该分布共由 $k$ 个混合成分组成，每个混合成分对应一个高斯分布。其中 $\mu_i$ 与 $\Sigma_i$ 是第 $i$ 个高斯混合成分的参数，而 $\alpha_i > 0$ 为相应的 “混合系数(mixture coefficient)” ，且有 $\sum^k_{i=1} \alpha_i = 1$ 。

假设样本的生成过程由高斯混合分布给出：首先，根据 $\alpha_1, \alpha_2, \ldots, \alpha_k$ 定义的先验分布选择高斯混合成分，其中 $\alpha_i$ 为选择第 $i$ 个混合成分的概率；然后，根据被选择的混合成分的概率密度函数进行采样，从而生成相应的样本。

若训练集 $D = \{x_1, \ldots, x_m \}$ 由上述过程生成，令随机变量 $z_j \in \{1, 2, \ldots, k \}$ 表示生成样本 $x_j$ 的高斯混合成分，其取值未知。显然，$z_j$ 的先验概率 $P(z_j = i)$ 对应于 $\alpha_i (i = 1, 2, \ldots, k)$ 。根据贝叶斯定理，$z_j$ 的后验概率分布对应于
$$
\begin{eqnarray}
p_{\mathcal{M}}(z_j = i | x_j)
&=& \frac{P(z_j = i) \cdot p_{\mathcal{M}}(x_j | z_j = i)}{p_{\mathcal{M}}(x_j)} \\
\\
\tag{9.30} \label{eq_bayes_posterior}
&=& \frac{\alpha_i \cdot p(x_j | \mu_i, \Sigma_i) }{\sum^k_{l=1} \alpha_l \cdot p(x_j | \mu_l, \Sigma_l)} .
\end{eqnarray}
$$
换言之，$p_{\mathcal{M}}(z_j = i | x_j)$ 给出了样本 $x_j$ 由第 $i$ 个高斯混合成分生成的后验概率。为方便叙述，将其简记为 $\gamma_{ji} (i = 1, 2, \ldots, k)$ 。

当高斯混合分布(式9.29)已知时，高斯混合聚类将把样本集 $D$ 划分为 $k$ 个簇 $\mathcal{C} = \{C_1, \ldots, C_k \}$ ，每个样本 $x_j$ 的簇标记 $\lambda_j$ 如下确定：
$$
\tag{9.31} \label{eq_cluster_idx}
\lambda_j
= \underset{i \in \{1, 2, \ldots, k \}}{\operatorname{argmax}} \gamma_{ji} .
$$
因此，从原型聚类的角度来看，高斯混合聚类是采用概率模型(高斯分布)对原型进行刻画，簇划分则由原型对应后验概率确定。

那么，对于式(9.29)，模型参数 $\{\alpha_i, \mu_i, \Sigma_i | 1 \le i \le k \}$ 如何求解呢？显然，给定样本集 $D$ ，可采用极大似然估计，即最大化似然(对数似然)
$$
\begin{eqnarray}
LL(D)
&=& \ln \bigg(\prod^m_{j=1} p_{\mathcal{M}}(x_j) \bigg) \\
\\
\tag{9.32}
&=& \sum^m_{j=1} \ln \bigg(\sum^k_{i=1} \alpha_i \cdot p(x_j | \mu_i, \Sigma_i)  \bigg) ,
\end{eqnarray}
$$
常用 EM 算法[^8] 进行迭代优化求解，得到
$$
\tag{9.34}
\mu_i = \frac{\sum^m_{j=1} \gamma_{ji} x_j}{\sum^m_{j=1} \gamma_{ji}} ,
$$
即各**混合成分的均值**可通过样本加权平均来估计，样本权重是每个样本属于该成分的后验概率。类似的，对**混合成分的协方差矩阵**有
$$
\tag{9.35}
\Sigma_i = \frac{\sum^m_{j=1} \gamma_{ji}(x_j - \mu_i)(x_j - \mu_i)^{\mathsf{T}}}{\sum^m_{j=1} \gamma_{ji}} .
$$
对应混合系数 $\alpha_i$ ，除了要最大化 $LL(D)$ ，还需要满足 $\alpha_i \le 0,\sum^k_{i=1} \alpha_i = 1$ 。考虑解 $LL(D)$ 的拉格朗日形式
$$
\tag{9.36}
LL(D) + \lambda \bigg(\sum^k_{i=1} \alpha_i - 1 \bigg) ,
$$
其中 $\lambda$ 为拉格朗日乘子。由式(9.36)对 $\alpha_i$ 的导数为 0，有
$$
\tag{9.37}
\sum^m_{j=1} \frac{p(x_j | \mu_i, \Sigma_i)}{\sum^k_{l=1} \alpha_l \cdot p(x_j | \mu_l, \Sigma_l)} + \lambda = 0 ,
$$
两边同乘以 $\alpha_i$ ，对所有样本求和可知 $\lambda = -m$ ，有
$$
\tag{9.38}
\alpha_i = {1 \over m} \sum^m_{j=1} \gamma_{ji} ,
$$
即每个**高斯成分的混合系数**由样本属于该成分的平均后验概率确定。

由上述推导即可获得高斯混合模型的 EM 算法：在每步迭代中，先根据当前参数来计算每个样本属于每个高斯混合成分的后验概率 $\gamma_{ji}$ （E 步），再根据式(9.34)、(9.35)和(9.38)更新模型参数 $\{\alpha_i, \mu_i, \Sigma_i | 1 \le i \le k \}$ （M 步）。

---

`高斯混合聚类算法` 

---

**输入**：样本集 $D = \{x_1, \ldots, x_m \}$ ；

​			高斯混合成分个数 $k$ 。

**过程**：

01: 初始化高斯混合分布的模型参数 $\{\alpha_i, \mu_i, \Sigma_i | 1 \le i \le k \}$ 

02: **repeat**

03: 		**for** $j = 1, 2, \ldots, m$ **do**

04: 		根据式($\ref{eq_bayes_posterior}$)计算样本 $x_j$ 由各混合成分生成的后验概率，即 $\gamma_{ji} = p_{\mathcal{M}}(z_j = i | x_j) (1 \le i \le k)$ 

05: 		**end for** 

06: 		**for** $i = 1, 2, \ldots, k$ **do** 

07: 				计算新的均值向量：$\mu_i' = \frac{\sum^m_{j=1} \gamma_{ji} x_j}{\sum^m_{j=1} \gamma_{ji}}$ 

08: 				计算新的协方差矩阵：$\Sigma_i' = \frac{\sum^m_{j=1} \gamma_{ji}(x_j - \mu_i)(x_j - \mu_i)^{\mathsf{T}}}{\sum^m_{j=1} \gamma_{ji}}$ 

09: 				计算新的混合系数：$\alpha_i' = {1 \over m} \sum^m_{j=1} \gamma_{ji}$ 

10: 		**end for** 

11: 		将模型参数 $\{\alpha_i, \mu_i, \Sigma_i | 1 \le i \le k \}$ 更新为 $\{\alpha_i', \mu_i', \Sigma_i' | 1 \le i \le k \}$ 

12: **until** 满足停止条件

13: $C_i = \phi (1 \le i \le k)$ 

14: **for** $j = 1, 2, \ldots, m$ **do** 

15: 		根据式($\ref{eq_cluster_idx}$)确定 $x_j$ 的簇标记 $\lambda_j$ ;

16: 		将 $x_j$ 划入相应的簇：$C_{\lambda_j} = C_{\lambda_j} \cup \{x_j \}$ 

17: **end for** 

**输出**：簇划分 $\mathcal{C} = \{C_1, C_2, \ldots, C_k \}$ 

---



[^7]: $p_{\cal{M}} (·)$ 也是概率密度函数，$\int {p_{\cal{M}}(x)dx} = 1$.
[^8]: Expectation-Maximization 算法(EM，期望最大化算法) 是常用的估计参数隐变量的利器，它是一种迭代式的方法，其核心思想是：若模型参数 $\Theta$ 已知，则可根据训练数据推断出最优隐变量 $\mathbf{Z}$ 的值（E 步）；反之，若 $\mathbf{Z}$ 的值已知，则可方便地对参数 $\Theta$ 做极大似然估计（M 步）。进一步，若我们不是取 $\mathbf{Z}$ 的期望，而是基于$\Theta$ 计算隐变量 $\mathbf{Z}$ 的概率分布 $P(\bf{Z} | X, \Theta)$ ，则 EM 算法的两个步骤是：以当前参数 $\Theta^t$ 推断 $P(\bf{Z} | X, \Theta^t)$ ，并计算对数似然 $LL(\bf{\Theta} | X, Z)$ 关于 $\bf{Z}$ 的期望，即 $\mathbb{E}(\Theta | \Theta^t)$（E 步）；寻找参数最大化期望似然，即 $\Theta^{t+1} = \text{argmax}_{\Theta} \mathbb{E}(\Theta | \Theta^t)$ （M 步）。EM 算法可看作用 “坐标下降法” 来最大化对数似然下界的过程。



### Gaussian Mixture Models in Action

A *Gaussian mixture model (GMM)* is a probabilistic model that assumes that the instances were generated from  a mixture of several Gaussian distributions whose parameters are unknown. All the instances generated from a single Gaussian distribution from a cluster that typically looks like an ellipsoid with different shape, sizes, density and orientation.

There are several GMM variants. In the simplest variant, implemented in the `GaussianMixture` class, you must know in advance the number $k$ of Gaussian distributions. The dataset $\bf{X}$ is assumed to have been generated through the following probabilistic process:

- For each instance, a cluster is picked randomly from among $k$ clusters. The probability of choosing the $j^{th}$ cluster is defined by the cluster's weight, $\phi^{(j)}$. The index of the cluster chosen for the $i^{th}$ instance is noted as $z^{(i)}$ .
- If $z^{(i)} = j$, meaning the $i^{th}$ instance has been assigned to the $j^{th}$ cluster, the location $\bf{x}^{(i)}$ of this instance is sampled randomly from the Gaussian distribution with mean $\mathbf{\mu}^{(j)}$ and covariance matrix $\mathbf{\Sigma}^{(j)}$ . This is noted $\mathbf{x}^{(i)} \sim \mathcal{N}(\mathbf{\mu}^{(j)}, \mathbf{\Sigma}^{(j)})$.

This generative process can be represented as a graphical model (Figure 9-16).

![gmm](./images/handson_cluster_gmm.png)



#### GMM for Clustering

So, what can you do with such a model? Well, given the dataset $\bf{X}$ , you typically want to start by estimating the weights $\phi$ and all the distribution parameters $\mathbf{\mu}^{(1)}$ to $\mathbf{\mu}^{(k)}$ and $\mathbf{\Sigma}^{(1)}$ to $\mathbf{\Sigma}^{(k)}$ . Sklearn's `GaussianMixture` class makes this super easy:

```python
from sklearn.mixture import GaussianMixture

# This class relies on the Expectation-Maximization(EM) algorithm,
# which has many similarities with K-Means algorithm:
# it also initializes the cluster parameters randomly,
# then it repeats two steps until convergence:
# 	* first assigning instances to clusters (E step)
# 	* then updating the clusters (M step)
# Think of EM as a generalization of K-Means that not only finds
# 	* the clusters (mu_1 to mu_k), but also
# 	* their size, shape, and orientation (Sigma_1 to Sigma_k), as well as
# 	* their relative weights (phi_1 to phi_k)
# Unlike K-Means, EM uses soft clustering assignments, not hard assignments,
# unfortunately, just like K-Means, EM can end up converging to poor solutions,
# so it needs to be run several times, keeping only the best solution. This is
# why we set n_init=10. (By default, n_init=1)

gmm = GaussianMixture(n_components=3, n_init=10)
gmm.fit(X)

print(gmm.converged_) 		# True or False
print(gmm.n_iter_) 			# how many EM iterations using
print(gmm.weights_)			# cluster weights
print(gmm.means_) 			# means vectors
print(gmm.covariances_) 	# covariance matrices

# now that the gmm can easily
# 1. assign each instance to the most likely cluster (hard clustering)
# 2. estimate the probability that it belongs to a particular cluster (soft clustering)
res_hc = gmm.predict(X) 			# hard clustering
res_sc = gmm.predict_proba(x) 		# soft clustering

# estimate the density of the model at any given location
log_pdf_scores = gmm.score_samples(X)
pdf_values = np.exp(log_pdf_scores)
# these pdf_values are not probabilities, but probability densities,
# to estimate the probability that an instance will fall within a
# particular region, one would have to integrate the PDF over that region.

# A GMM is a generative model, meaning you can sample new instances form it
# (note that they are ordered by cluster index):
X_new, y_new = gmm.sample(6)
```



Figure 9-17 shows the cluster means, the decision boundaries (dashed lines), and the density contours of this model.

![trained gmm](./images/handson_trained_gmm.png)

It seems the algorithm clearly found an excellent solution. Of course, we made its task easy by generating the data using a set of 2D Gaussian distributions (real life data is not always so Gaussian and low-dimensional). We also gave the algorithm the correct number of clusters.

When there are many dimensions, or many clusters, or few instances, EM can struggle to converge to the optimal solution. In such cases, we might need to reduce the difficulty of the task by limiting the number of parameters that the algorithm has to learn. One way to do this is to constraints the covariance matrices (limited the range of shapes and orientations the clusters can have) by setting the `covariance_type` hyperparameter to one of the following values:

- `covariance_type="spherical"` : 

  All clusters must be spherical, but can have different diameters (i.e., different variances)

- `covariance_type="diag"` : 

  Clusters can take on any ellipsoidal shape of any size, but ellipsoid's axes must parallel to the coordinate axes

- `covariance_type="tied"` : 

  All the cluster must have the same ellipsoidal shape, size, and orientation (i.e., all share one covariance matrix)

- `covariance_type="full"` :  (by default)

  This means that each cluster can take on any shape, size, and orientation. If there is a large numbers of features, it will not scale well.



![constrained gmm](./images/handson_constrained_gmm.png)



#### GMM for Anomaly Detection

*Anomaly detection* (a.k.a., *outlier detection*) is the task of detecting instances that deviate strongly from the norm. Using GMM for anomaly detection is quite simple: any instance located in a low-density region can be considered an anomaly. So one must define what density threshold to use.

For example, in a manufacturing company that tries to detect defective products, the ratio of defective products is usually well known. Say it is equal to 4%. You then set the density threshold ($\rho$) to be the value that results in having 4% of the instances located in areas below $\rho$:

- If getting too many false positives (good products flagged as defective), lower the value of $\rho$ 
- If getting too many false negatives (defective products  not flag as defective), lower the value of $\rho$ 

This is the usual precision/recall trade-off[^9] .

```python
# defective products example

densities = gmm.score_samlpe(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
```

![anomaly detection](./images/handson_gmm_anomaly_detection.png)

[^9]: Evaluate model performance in classification task.



#### Selecting the Number of Clusters

With K-Means, we could use the inertia or the silhouette score to select the appropriate number of clusters. But with GMM, it is not possible to use these metrics because they are not reliable when the clusters are not spherical or have different sizes. Instead, we can try to find the model that minimizes a *theoretical information criterion*, such the *Bayes information criterion (BIC)* or the *Akaike information criterion (AIC)*, defined as follows
$$
\begin{eqnarray}
BIC &=& \log(m)p - 2 \log(\hat{L}) \\
\\
AIC &=& 2p - 2 \log(\hat{L})
\end{eqnarray}
$$
where $m$ is the number of instances, $p$ is the number of parameters learned by the model, and $\hat{L}$ is the maximized value of the *likelihood function* of the model.

Both the $BIC$ and $AIC$ penalize models that have more parameters to learn (e.g., more clusters) and reward models that fit the data well. They often end up selecting the same model. When they differ, $BIC$ tends to select simpler model (fewer parameters) while not fit the data quite as well as $AIC$ (especially true for larger datasets).

To compute the $BIC$ and $AIC$ , call the `bic()` and `aic()` methods:

```python
# gmms contains gmm trained with different k
for gmm in gmms:
    bic_scores.append(gmm.bic(x))
    aic_scores.append(gmm.aic(x))
```

![bic aic metrics](./images/handson_gmm_bic_aic_metrics.png)















