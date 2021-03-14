# Tree-based models

## Part-I: Theorist views

**基本术语和符号约定**

一般地，令 $D = \{x_1, x_2, \ldots, x_m \}$ 表示包含 $m$ 个示例的数据集，每个示例由 $d$ 个属性描述，则每个示例 $x_i = (x_{i1}, x_{i2}, \ldots, x_{id})$ 是 $d$ 维样本空间 $\mathcal{X}$ 的一个向量[^1]，$x_i \in \mathcal{X}$, 其中 $x_{ij}$ 是 $x_i$ 在第 $j$ 个属性上的取值， $d$ 称为样本 $x_i$ 的“维数”（dimensionality）。

要建立一个关于“预测(prediction)”的模型，单有示例数据（也称为样本，sample）还不行，我们还需要获得训练样本的“结果”信息，例如，一个描述西瓜的记录“（（色泽=青绿；根蒂=蜷缩；敲声=浊响），好瓜）”。这里，关于示例结果的信息，例如 “好瓜” ，称为 “标记(label)”；拥有了标记信息的示例，则称之为 "样例(example)"。

一般地，用 $(x_i, y_i)$ 表示第 $i$ 个样例，其中 $y_i \in \mathcal{Y}$ 是示例 $x_i$ 的标记， $\mathcal{Y}$ 是所有标记的集合，亦称“标记空间(label space)”或“输出空间”。

如果我们想要预测的是离散值，例如 “好瓜” “坏瓜”，此类学习任务称为 “分类(classification)”；如果要预测的是连续值， 例如西瓜的成熟度0.9，0.4，此类学习任务称为 “回归(regression)”。二分类(binary classification)任务中，通常令 $\mathcal{Y} = \{-1, +1 \}$ 或  $\mathcal{Y} = \{0, 1 \}$；对于多分类(multi-class classification), $|\mathcal{Y}| > 2$；对回归任务，$\mathcal{Y} = \R$，$\R$ 为实数集。



[^1]: 由 $d$ 个属性张成的 $d$ 维空间中，每个示例都可以在这个空间中找到自己的坐标位置，每个空间中的点对应一个坐标向量，因此：一个示例就是一个“特征向量”（feature vector）。



## Decision Tree

### 决策树生成算法

一般的，一棵决策树包含一个根结点、若干个内部结点和若干个叶结点；叶结点对应于决策结果，其他每个结点对应于一个属性测试；每个结点包含的样本集合根据属性测试的结果被划分到子结点中；根结点包含样本全集。从根结点到每个叶结点的路径对应了一个判定测试序列。

决策树学习的目的是为了产生一棵泛化性能强的决策树，亦即处理未见示例（unseen samples）的能力强的决策树。其基本流程遵循简单且直观的“分而治之”（divide-and-conquer）策略，如`决策树学习基本算法`所示。

---

`决策树学习基本算法`

---

**输入**： 训练集 $D = \{(x_1, y_1), \dots, ({x_m, y_m}) \}$;

​			属性集 $A = \{a_1, \ldots, a_d \}$

**过程**： 函数 $\text{TreeGenerate}(D, A)$

1: 生成结点 $\text{node}$;

2: **if** $D$ 中样本全属于同一类别 $C$ **then**

3: 	将 $\text{node}$ 标记为 $C$ 类叶结点；**return**

4: **end if**

5: **if** $A = \phi$ **OR** $D$ 中样本在$A$ 上取值相同 **then**

6: 	将 $\text{node}$ 标记为叶结点，其类别标记为 $D$ 中样本数量最多的类；**return**

7: **end if**

**8**: 从 $A$ 中选择最优划分属性 $a_*$;

9: **for** 属性 $a_*$ 的每一个值 $a_*^v$ **do**

10: 	为 $\text{node}$ 生成一个分支；令 $D_v$ 表示 $D$ 中在 $a_*$ 上取值为 $a_*^v$ 的样本子集；

11: 	**if**  $D_v$ 为空 **then**

12: 		将分支结点标记为叶结点，其类别标记为 $D$ 中样本数量最多的类；**return**

13: 	**else**

14: 		以 $\text{TreeGenerate}(D_v, A - a_*)$ 为分支结点

15: 	**end if**

16: **end for**

**输出**： 以 $\text{node}$ 为根结点的一棵决策树

---

显然，决策树的生成时一个递归过程，在`决策树基本算法`中，有三种情形会导致递归返回：

1. 当前结点包含的样本全部属于同一类别 （无需进一步划分）
2. 当前属性集为空，或是所有样本在所有属性上取值相同 （无法进一步划分）
3. 当前结点包含的样本集合为空 （不能进一步划分）

在第2种情形下，我们把当前结点标记为叶结点，并将其类别设定为该结点中样本数量最多的类别；在第3种情形下，同样把当前结点标记为叶结点，但将其类别设定为其父结点所含样本最多的类别，注意这两种情形处理实质不同：情形2中是利用当前结点的后验分布，而情形3中则是把父结点的样本分布作为当前结点的先验分布。

> 《The hundred-Page Machine Learning》
>
> ![build tree the 1st split](./images/looPagesML_dtree_build.png)
>
> The ID3 learning algorithm works as follows. Let $\cal{S}$ denotes a set of labeled examples. In the begining, the decision tree only has a start noed (root node) that contains all examples: $\mathcal{S} = \{(\mathbb{x}_i, y_i) \}^N_i$. Start with a constant model $f_{ID3}$ :
> $$
> \tag{6}
> f_{ID3} = {1 \over |\mathcal{S}|} \sum_{(\mathbb{x},y) \in \mathcal{S}} y .
> $$
> The prediction given by the above model, $f_{ID3}(\mathbb{x})$, would be the same for any input $\mathbb{x}$. The corresponding decision tree is shown in fig4(a).
>
> The we search through all features $j = 1, \ldots, D$ and all thresholds $t$, and split the set $\cal{S}$ into two subsets:
>
> - $\mathcal{S}_{\_} = \{(\mathbb{x},y) | (\mathbb{x},y) \in \mathcal{S}, x^{(j)} < t \}$ and
> - $\mathcal{S}_{+} = \{(\mathbb{x},y) | (\mathbb{x},y) \in \mathcal{S}, x^{(j)} \ge t \}$ .
>
> The new two subsets would go to two new leaf nodes (or inter nodes), and we evaluate, for all possible pairs $(j, t)$ how good the split with pieces $\mathcal{S}_{\_}$ and $\mathcal{S}_{+}$ is (see the followed section [划分选择](###划分选择)). Finally, we pick the best such values $(j, t)$ for splitting $\cal{S}$ into  $\mathcal{S}_{\_}$ and $\mathcal{S}_{+}$ , from two new leaf nodes, and continue recursively on  $\mathcal{S}_{\_}$ and $\mathcal{S}_{+}$ (or quit if reach some criterion). A decision tree after one split is illustrated in fig4(b).

### 划分选择

决策树学习的关键是如何选择最优划分属性（`决策树基本算法` 第8行）。一般而言，随着划分过程不断进行，我们希望决策树的分支结点所包含的样本尽可能属于同一类别，即结点的“纯度”（purity）越来越高。

#### 信息增益

“信息熵”（information entropy）是度量样本集合纯度最常用的一种指标。假定当前样本集合 $D$ 中第 $k$ 类样本所占的比例为 $p_k \ (k=1, \ldots, |\mathcal{Y}|)$，则 $D$ 信息熵定义为
$$
\tag{4.1}
\text{Ent}{(D)} = - \sum^{|\mathcal{Y}|}_{k=1} p_k \text{log}_2 p_k.
$$
$\text{Ent}(D)$ 的值越小，则 $D$ 的纯度越高。

假定离散属性 $a$ 有 $V$ 个可能的取值 $\{a^1, \ldots, a^V \}$，若使用 $a$ 来对样本集 $D$ 进行划分，则会产生 $V$ 个分支结点，其中第 $v$ 个分支结点包含了 $D$ 中所有在属性 $a$ 上取值为 $a^v$ 的样本，记为 $D^v$。我们根据式(4.1)计算出 $D^v$ 的信息熵，再考虑到不同的分支结点所包含的样本数不同，给分支结点赋予权重 ${|D^v| \over |D|}$，即样本数越多的分支结点的影响越大，于是可以计算出用属性 $a$ 对样本集 $D$ 进行划分所获得的“信息增益(information gain)”
$$
\tag{4.2}
\text{Gain}(D, a) = \text{Ent}(D) - \sum^V_{v=1} {|D^v| \over |D|} \text{Ent}(D^v).
$$
一般而言，信息增益越大，则意味着使用属性 $a$ 来进行划分所获得的“纯度提升”越大。因此，我们可以用信息增益来进行决策树的划分属性选择，即选择属性 $a_* = argmax_{(a \in A)} \text{Gain}(D, a)$。

$\color{Green}{\bold{例子}}$

![xgs d2.0](./images/xgs_tree_dataset1.png)

以表4.1中的西瓜数据集2.0为例。该数据集包含17个训练样例，用以学习一棵能预测没有尝过的是不是好瓜的决策树。显然，分类的类别共两类（是好瓜，不是好瓜），$|\mathcal{Y}| = 2$。在决策树开始学习时，根结点包含 $D$ 中所有的样例，其中正例占 $p_1 = 8 / 17$ ，反例占  $p_1 = 9 / 17$。于是，根据式(4.1)可计算出根结点的信息熵为
$$
\text{Ent}(D) = - \sum^2_{k=1} p_k \text{log}_2 p_k = - \bigg({8 \over 17}\text{log}_2 {8 \over 17} + {9 \over 17}\text{log}_2 {9 \over 17} \bigg) \approx 0.998 .
$$
然后，我们要计算出当前属性集合｛色泽，根蒂，敲声，纹理，脐部，触感｝中每个属性的信息增益。以属性 “色泽” 为例，它有3个可能的取值：｛青绿，乌黑，浅白｝。若使用该属性对 $D$ 进行划分，则可得到3个子集，分别记为：$D^1 (色泽=青绿)，D^2 (色泽=乌黑)，D^3 (色泽=浅白）$。

由表4.1可得，子集 $D^1$ 包含编号为｛1，4，6，10，13，17｝的6个样例，其中正例占 $p_1 = 3 / 6$ ，反例占  $p_2 = 3 / 6$；子集 $D^2$ 包含编号为｛2，3，7，8，9，15｝的6个样例，其中正例占 $p_1 = 4 / 6$ ，反例占  $p_2 = 2 / 6$；子集 $D^3$ 包含编号为｛5，11，12，14，16｝的5个样例，其中正例占 $p_1 = 1 / 5$ ，反例占  $p_2 = 4 / 5$。根据式(4.1)可计算出用 “色泽” 划分之后所得到的3个分支结点的信息熵为
$$
\begin{eqnarray}
\text{Ent}(D^1) &=& - \bigg({3 \over 6}\text{log}_2 {3 \over 6} + {3 \over 6}\text{log}_2 {3 \over 6} \bigg) = 1.000, \\
\text{Ent}(D^2) &=& - \bigg({4 \over 6}\text{log}_2 {4 \over 6} + {2 \over 6}\text{log}_2 {2 \over 6} \bigg) = 0.918, \\
\text{Ent}(D^3) &=& - \bigg({1 \over 5}\text{log}_2 {1 \over 5} + {4 \over 5}\text{log}_2 {4 \over 5} \bigg) = 0.772, \\
\end{eqnarray}
$$
于是，根据式(4.2)可计算出属性 “色泽” 的信息增益为
$$
\begin{eqnarray}
\text{Gain}(D, 色泽)
&=& \text{Ent}(D) - \sum^3_{v=1} {|D^v| \over |D|} \text{Ent}(D^v) \\
&=& 0.998 - \bigg({6 \over 17} \times 1.000 + {6 \over 17} \times 0.918 + {5 \over 17} \times 0.772 \bigg) \\
&=& 0.109 .
\end{eqnarray}
$$
类似的，我们可以计算出其他属性的信息增益：
$$
\text{Gain}(D, 根蒂) = 0.143;\text{Gain}(D, 敲声) = 0.141 \\
\text{Gain}(D, 纹理) = 0.381;\text{Gain}(D, 脐部) = 0.289 \\
\text{Gain}(D, 触感) = 0.006.\qquad \qquad \qquad \qquad \quad \ \
$$
显然，属性 “纹理” 的信息增益最大，于是它被选为划分属性。图4.3给出了基于 “纹理” 对根结点进行划分的结果，各分支结点所包含的样例子集显示在结点中。

![tree first split](./images/xgs_tree1.png)

然后，决策树学习算法将对每个分支结点做进一步划分。以图4.3中第一个分支结点（“纹理=清晰”）为例，该结点包含的样例集合 $D^1$ 中有编号为 ｛1，2，3，4，5，6，8，10，15｝的9个样例，可用属性集合为 ｛色泽，根蒂，敲声，脐部，触感｝[^2]。基于 $D^1$ 计算出各个属性的信息增益：
$$
\text{Gain}(D, 根蒂) = 0.458;\text{Gain}(D, 敲声) = 0.331 \\
\text{Gain}(D, 色泽) = 0.043;\text{Gain}(D, 脐部) = 0.458 \\
\text{Gain}(D, 触感) = 0.458.\qquad \qquad \qquad \qquad \quad \ \
$$
“根蒂”、“脐部”、“触感” 3个属性均取得最大的信息增益，可任选其中之一作为划分属性。类似的，对每个分支结点进行上述操作，最终得到的决策树如图4.4所示。

![tree first split](./images/xgs_tree2.png)

[^2]:  上一步的划分属性“纹理”，不再作为候选划分属性。

#### 增益率

在上面的例子中，我们有意忽略了表4.1中的 “编号” 这一列。如果把 “编号” 也作为一个候选划分属性，则根据式(4.2)可计算出它的信息增益为$0.998$，远大于其他候选划分属性。这很容易理解：“编号” 将产生17个分支，每个分支结点仅包含一个样本，这些分支结点的纯度已达到最大。然而，这样的决策树显然不具有泛化能力，无法对新样本进行有效预测。

实际上，信息增益准则对可取值数目较多的属性有所偏好。为减少这种偏好可能带来的不利影响，C4.5决策树算法使用 “增益率(gain ratio)” 来选择最优划分属性。采用与式4.2相同的符号表示，增益率定义为
$$
\tag{4.3}
\text{Gain_ratio}(D, a) = \frac{\text{Gain}(D, a)}{\text{IV}(a)},
$$
其中
$$
\tag{4.4}
\text{IV}(a) = - \sum^V_{v=1} {|D^v| \over |D|} \text{log}_2 {|D^v| \over |D|}
$$
称为属性 $a$ 的 “固有值(intrinsic value)”。属性 $a$ 的可能取值数目越多（即 $V$ 越大），则 ${\text{IV}(a)}$ 的值也越大。

注：增益率准则对可取值数目较少的属性有所偏好，因此，C4.5算法并不是直接选择增益率最大的候选划分属性，而是使用一个**启发式**：先从候选划分属性中找出 *信息增益* 高于平均水平的属性，再从中选择 *增益率* 最大的。

#### 基尼指数

CART(Classification and Regression Tree) 决策树使用基尼指数（Gini index）来选择划分属性。数据集D的纯度可用基尼指数来度量：
$$
\begin{eqnarray}
Gini(D) 
&=& \sum^{|\mathcal{Y}|}_{k=1} \sum_{k' \ne k} p_k p_{k'} \\
&=& 1 - \sum^{|\mathcal{Y}|}_{k=1} {p_{k}}^2.
\end{eqnarray}
$$
直观来说，$Gini(D)$反映了从数据集$D$中随机抽取两个样本，其类别标记不一致的概率。因此$Gini(D)$越小，则数据集$D$的纯度越高。

属性 $a$ 的基尼指数定义为
$$
\text{Gini_index}(D, a) = \sum^{V}_{v=1} \frac{|D^v|}{|D|}Gini(D^v)
$$
于是，我们候选属性集合$A$中，选择那个使得划分后基尼指数最小的属性作为最优划分属性，即$a^* = argmin_{(a \in A)} \text{Gini_index}(D, a)$.

> 《hands-on Machine Learning with sklearn, Keras and tensorflow》
>
> The CART Training Algorithm
>
> **1. Classification Task** 
>
> **Sklearn** uses the CART algorithm to train Decision Tree (i.e., "growing" tree). The algorithm works by first splitting the training set into two subsets using a single feature $k$ and a threshold $t_k$ (e.g., "petal length $\le$ 2.45 cm"  which is a feature in iris data). How does it choose  $k$ and $t_k$ ? It searches for the pair ($k$, $t_k$) that produces the purest subsets (weighted by their size).
> $$
> \tag{6.2}
> J(k, t_k) = {m_{left} \over m} G_{left} + {m_{right} \over m} G_{right}
> $$
> where
>
> - $m_{left/right}$ is the number of instances in the left/right subset,
> - $G_{left/right}$ measures the impurity of the left/right subset.
>
> Equation 6.2 gives the cost function for classification task that the algorithm tries to minimize.
>
> Once the CART algorithm has successfully split the training set in two, it splits the subsets using the same logic, then the sub-subsets, and so on, recursively. It stops recursing once it reaches the maximum depth (`max_depth`), or if it cannot find a split that will reduce impurity. There are other additional stopping conditions hyperparameters such as `min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, and `max_leaf_nodes`. Increasing `min_*` hyperparameters or reducing `max_*` hyperparameters will regularize the model.
>
> **2. Regresssion Task** 
>
> The CART algorithm works mostly the same as earlier, except that instead of trying to split the training set in a way that minimizes impurity, it now tries to split the training set in a way that minimizes the MSE.
> $$
> \tag{6.2}
> J(k, t_k) = {m_{left} \over m} \text{MSE}_{left} + {m_{right} \over m} \text{MSE}_{right}
> $$
> where
>
> - $MSE_{node} = \sum_{i \in node} (\hat{y}_{node} - y^{(i)})$ ,
> - $\hat{y}_{node} = {1 \over m_{node}} \sum_{i \in node} y^{(i)}$
>
> 
>
> **3. Instability** 
>
> Decision Trees produce orthogonal decision boundaries (all splits are perpendicular to an axis), which makes them sensitive to trianing set rotation (The model on the right of figure 6-7 will not generalize well). Ony way to limit this problem is to use Principal Component Analysis (PCA), which often results in a better orientation of the training data.
>
> ![dtree instability](./images/hands-onML_dtree_instability.png)
>
> More generally, the main issue with Decision Trees is that they are very sensitive to small variations in the training data[^3]. Actually, since the training algorithm used by Sklearn is stochastic (means it randomly selects the set of features to evaluate at each node), it may produces very different models even on the same training data (unless you set the `random_state` hyperparameter).



[^3]: Which is mainly due to the nature of how decision tree growed using greedy algorithm and easily overfiting.



### 剪枝处理

剪枝 (pruning)是决策树学习算法对付 “过拟合” 的主要手段。在决策树学习过程中，为了尽可能正确分类训练样本，结点划分过程将不断重复，有时会造成决策树分支过多，以致于把训练集自身的一些特点当作所有数据都具有的一般性质从而导致过拟合。因此，通过主动去掉一些分支来降低过拟合的风险。

决策树剪枝的基本策略有 **预剪枝(prepruning)** 和 **后剪枝(post-pruning)** 。

预剪枝是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能的提升，则停止划分，并将当前结点标记为叶结点。

后剪枝则是先从训练集生成一棵完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能的提升，则将该子树替换为叶结点。

**如何判断决策树泛化性能是否提升呢？** 这可以使用常用的性能评估方法进行，如 “留出法”、“交叉验证法” 以及 “自助法” 等方法。

预剪枝会使得决策树的很多分支没有 “展开”，这不仅能降低过拟合的风险，还会显著减少训练和测试的时间开销。但另一方面，有些分支的当前划分虽不能提升泛化性能（甚至可能导致泛化性能暂时下降），但在其基础上进行的后续划分却有可能使得泛化性能显著提高；预剪枝基于 “贪心” 本质禁止这些分支展开，这给预剪枝决策树带来欠拟合的风险。

后剪枝决策树通常会比预剪枝决策树保留更多的分支。一般情形下，后剪枝决策树欠拟合风险很小，泛化性能往往优于预剪枝决策树。但后剪枝决策树的训练时间开销则大得多。

### 连续值属性和缺失值

**连续值处理**

由于连续属性的可取值数目不再有限，因此，不能直接根据连续属性的可取值来对结点进行划分。此时，**连续属性离散化**技术可派上用场。最简单的策略是采用二分法(bi-partition)对连续属性进行处理，这正是C4.5决策树算法中采用的机制。

给定样本集 $D$ 和连续属性 $a$，假定 $a$ 在 $D$ 上出现了 $n$ 个不同的取值，将这些取值从小到大进行排序，记为 {$a^1, \ldots, a^n$}。基于划分点 $t$ 可将 $D$ 分为子集 $D^-_t$ 和 $D^+_t$ ，其中 $D^-_t$ 包含哪些在属性 $a$ 上取值不大于 $t$ 的样本，而 $D^+_t$ 则包含那些大于 $t$ 的样本。显然，对相邻的属性取值 $a^i$ 与 $a^{i+1}$ 来说，$t$ 在区间 [$a^i, a^{i+1}$) 中取任意值所产生的划分结果相同。因此，对连续属性 $a$，我们可考察包含 $n - 1$ 个元素的候选划分点集合
$$
\tag{4.7}
T_a = \bigg\{{a^i + a^{i+1} \over 2} | 1 \le i \le n-1 \bigg\},
$$
即把区间 [$a^i, a^{i+1}$) 的中位点作为候选划分点[^4]。然后我们就可以像离散属性值一样来考虑这些划分点，选取最优的划分点进行样本集合的划分。例如，可对式4.2稍加改造：
$$
\begin{eqnarray}
\text{Gain}(D, a) 
&=& \text{max}_{(t \in T_a)} \text{Gain}(D, a, t) \\
\tag{4.8}
&=& \text{max}_{(t \in T_a)} \text{Ent}(D) - \sum_{\lambda \in \{-, + \}} {|D^{\lambda}_t| \over |D|} \text{Ent}(D^{\lambda}_t),
\end{eqnarray}
$$
其中，$\text{Gain}(D, a, t)$ 是样本集 $D$ 基于划分点 $t$ 二分后的信息增益。于是，我们就可选择使 $\text{Gain}(D, a, t)$ 最大化的划分点。



[^4]: 可将划分点设为该属性在训练集中出现的不大于中位点的最大值。由此，决策树使用的划分点都出现在训练集中。



**缺失值处理**

现实任务中常会遇到不完整样本，即样本的某些属性值缺失。在属性数目较多的情形下，往往会有大量样本出现缺失值。如果简单地放弃不完整样本，仅使用无缺失值的样本进行学习，显然是对数据信息的极大浪费。

我们需要解决两个问题：

1. 如何在属性值缺失的情况下进行划分属性的选择？
2. 给定划分属性，如果样本在该属性上的值缺失，如何对样本进行划分？



> The real handling approaches to missing data does not use data point with missing values in the evaluation of a split. However, when child nodes are created and trained, those instances are distributed somehow.
>
> I know about the following approaches to distribute the missing value instances to child nodes:
>
> - simply ignoring the missing values (like ID3 and other old algorithms does) or treating the missing values as another category (in case of a nominal feature). Those approachs were used in the early stages of decision tree development.
>
> - all goes to the node which already has the biggest number of instances (CART, but not its primary rule)
> - distribute to all children, but with diminished weights, proportional with the number of instances from each child node (C4.5 and others)
> - distribute randomly to only one single child node, eventually according with a categorical distribution (various implementations of C4.5 and CART for faster funing time)
> - build, sort and use surrogates to distribute instances to a child node, where surrogates are input features which resembles best how the test feature send data instances to left or right child node (CART, if that fails, the majority rule is used)
>
> This answer was copied from [here](https://stats.stackexchange.com/questions/96025/how-do-decision-tree-learning-algorithms-deal-with-missing-values-under-the-hoo).



### 多变量决策树

如果我们把每个属性视为坐标空间中的一个坐标轴，则 $d$ 个属性描述的样本就对应了 $d$ 维空间中的一个数据点，对样本分类意味着在这个坐标空间中寻找不同样本之间的分类边界。

决策树所形成的分类边界有一个明显的特点：轴平行(axis-parallel)，即它的分类边界由若干个与坐标轴平行的分段组成。

![data3a](./images/xgs_tree_dataset3a.png)

以表4.5中的西瓜数据$3.0 \alpha$为例，将它作为训练集学习得图4.10所示的决策树，其分类边界如图4.11所示。

![tree3a](./images/xgs_tree3a2.png)

显然，分类边界的每一段都是与坐标轴平行的。这样的分类边界使得学习结果有较好的可解释性，因为每一段划分都直接对应了某个属性取值。但在学习任务的真实分类边界比较复杂时，必须使用很多段划分才能获得较好的近似，如图4.12所示；此时的决策树会相当复杂，由于需要进行大量属性测试，预测时间开销会很大。

![tree3a3](./images/xgs_tree3a3.png)

如果能够使用斜的划分边界，如图4.12中的红色线段所示，则决策树模型将大为简化。

**“多变量决策树”(multivariate decision tree)** 就是能实现这样的 “斜划分” 甚至更复杂划分的决策树。以实现斜划分的多变量决策树为例，在此类决策树中，非叶结点不再是仅对某个属性，而是对属性的线性组合进行测试；换言之，每个非叶结点是一个形如 $\sum^d_{i=1} w_i a_i = t$ 的线性分类器，其中 $w_i$ 是属性 $a_i$ 的权重， $w_i$ 和 $t$ 可以在该结点所含的样本集和属性集上学得[^6]。于是，与传统的 “单变量决策树(univariate decision tree)” 不同，在多变量决策树的学习过程中，不是为每个非叶结点寻找一个最优划分属性，而是试图建立一个合适的线性分类器。

例如对西瓜数据$3.0 \alpha$，我们可以学得图4.13这样的多变量决策树，其分类边界如图4.14所示。

![tree3a4](./images/xgs_tree3a4.png)

### 阅读材料

在**信息增益、增益率、基尼指数**之外，人们还设计了许多其他的准则用于决策树划分选择，然而有实验研究表明[^7]，这些准则虽然对决策树的尺寸有较大影响，但对泛化性能的影响很有限；对信息增益和基尼指数进行的理论分析[^8]也显示出，它们仅在 $2\%$ 的情况下会有所不同。而剪枝方法和剪枝程度对决策树的泛化性能影响显著，有实验研究[^9]表明，在数据带有噪声时，通过剪枝甚至可将决策树的泛化性能提高 $25\%$。

多变量决策树算法主要有 $OC1$[^10]，$OC1$ 算法先贪心地寻找每个属性的最优权值，在局部优化的基础上再对分类边界进行随机扰动以试图找到更好的边界；Brodley and Utgoff[^11] 则直接引入了线性分类器学习的最小二乘法。还有一些算法试图在决策树的叶结点上嵌入神经网络，以结合这两种学习机制的优势，例如 “感知机树(Perceptron tree)”[^12] 在每个叶结点上训练一个感知机，也有直接在叶结点上嵌入多层神经网络的模型[^13]。

有一些决策树学习算法可进行 “增量学习(incremental learning)”，即在接收到新样本后可对已学得的模型进行调整，而不用完全重新学习。主要机制是通过调整分支路径上的划分属性次序来对树进行部分重构，代表性算法有ID4[^14]、ID5R[^15]、ITI[^16]等。增量学习可有效降低每次接收到新样本后的训练时间开销，但多步增量学习后的模型会与基于全部数据训练而得的模型有较大差别。



[^6]: 待补充。
[^7]: [Mingers, 1989b]
[^8]: [Raileanu and Stoffel, 2004]
[^9]: [Mingers, 1989a]
[^10]: [Mruthy et al., 1994]
[^11]: [Brodley and Utgoff, 1995]
[^12]: [Utgoff, 1989b]
[^13]: [Guo and Gelfand, 1992]
[^14]: [Schlimmer and Fisher, 1986]
[^15]: [Utgoff, 1989a]
[^16]: [Utgoff et al., 1997]



## 随机森林(Random Forest)

根据个体学习器的生成方式，目前集成学习[^17]方法大致可分为两大类，

- 个体学习器之间存在强依赖关系、必须串行生成的序列化方法，代表算法Boosting[^18];
- 个体学习器之间不存在强依赖关系、可同时生成的并行化方法，代表算法Bagging和 “随机森林”.

想要得到泛化性能强的集成，集成中的个体学习器应尽可能相互独立；虽然 “独立” 在现实任务中无法做到，但可以设法使基学习器尽可能具有较大的差异。

给定一个训练数据集，一种可能的做法是对训练样本进行采样，产生出若干个不同的子集，再从每个数据子集中训练出一个基学习器。这样，由于训练数据不同，我们获得的基学习器可望具有比较大的差异。然而，为获得好的集成，我们同时还希望个体学习器不能太差。如果采样出的每个子集都完全不同，则意味着每个基学习器只用到了一小部分训练数据，甚至可能不足以进行有效学习，这就无法保证产出比较好的基学习器。为解决这个问题，我们可考虑使用互相有交叠的采样子集。

![hard voting classifier](./images/hands-onML_ensemble_majority_vote.png)

*Figure 7-2. Hard voting classifier predictions. Copy from the book《hands-on Machine Learning with sklearn, Keras and tensorflow》*



### BAGGING (Bootstrap AGGregatING)

Bagging 是并行式集成学习方法最著名的代表。从名字即可看出，它直接基于自助采样法[^19]。因此，我们知道初始训练集中约有 $63.2\%$ 的样本出现在采样集中。我们可以采样出 $T$ 个含 $m$ 个训练样本的采样集，然后基于每个采样集训练出一个基学习器，再将这些基学习器进行结合。这就是Bagging的基本流程。

在对预测输出进行结合时，Bagging通常对分类任务使用简单投票法，对回归任务采用简单平均法。

---

Bagging 算法

---

**输入**:  训练集 $D = \{(x_1, y_1), \ldots, (x_m, y_m) \}$;

​			基学习算法 $\mathcal{L}$;

​			训练轮数 $T$.

**过程**：

1: **for** $t = 1, \ldots, T$ **do**

2: 	$h_t = \mathcal{L} (D, D_{bs})$

3: **end for**

**输出**: $H(x) = argmax_{(y \in \mathcal{Y})} \sum^T_{t=1} \mathbf{I}(h_t(x) = y)$

---

其中，$D_{bs}$ 是自助采样产生的样本分布。

从偏差-方差分解的角度看，Bagging主要关注降低方差，因此因此它在不剪枝决策树、神经网络等易受到样本扰动的学习器上效用更为明显。

> 《hands-on Machine Learning with sklearn, Keras and tensorflow》
>
> **Bagging and Pasting** 
>
> One way to get a diverse set of classifiers is to use very different training algorithms (such as SVMs, LR, DTs etc). Another approach is to use the same training algorithm for every predictor and train them on different random subsets of the training set. When sampling is performed **with replacement**, this method is called **bagging**, when sampling is preformed **without replacement**, it is called **pasting**.
>
> In other words, both bagging and pasting allow training instances to be sampled several times across multiple predictiors, but only bagging allows training instances to be sampled several times for the same predictor. This sampling and training process is represented in Figure 7-4.
>
> ![bagging pasting](./images/hands-onML_ensemble_bagging_pasting.png)
>
> Once all predictors are trained, the ensemble can make a prediction for a new instance by simply aggregating the predictions all predictors. The aggregation function is typically
>
> - the *statistical mode*[^20] for classification, or
> - the *statistical average* for regression.
>
> Generally, the net result is that the ensemble has a similar bias but a lower variance than single predictor trained on the original train set.

### 随机森林 RF

随机森林是Bagging的一个扩展变体。RF在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择。具体来说，传统决策树在选择划分属性时是在当前结点的属性集（假定有 $d$ 个属性）中选择一个最优属性；而在 RF 中，对基决策树的每个结点，先从该结点的属性集中随机选择一个包含 $k$ 个属性的子集，然后再从这个子集中选择一个最优属性用于划分。这里的参数 $k$ 控制了随机性的引入程度：

- $k = d$, 则基决策树的构建与传统决策树相同；
- $k = 1$, 则是随机选择一个属性用于划分；一般推荐 $k = \text{log}_2 d$.

可以看出，随机森林对Bagging只做了小改动，但是与Bagging中基学习器的 “多样性” 仅通过样本扰动而来不同，随机森林中基学习器的多样性不仅来自样本扰动，还来自属性扰动，这就使得最终集成的泛化性能可通过个体学习器之间差异程度的增加而进一步提升。

值得一提的是，随机森林的训练效率通常优于Bagging，因为在个体决策树的构建过程中，Bagging 使用的是 “确定型” 决策树，在选择划分属性时要对结点的所有属性进行考察，而随机森林使用的 “随机型” 决策树则只需考察一个属性子集。

> 《hands-on Machine Learning with sklearn, Keras and tensorflow》
>
> **1. Random Forest**  
>
> A Random Forest is an ensemble of Decision Trees, generally trained via the bagging method (or sometimes pasting), typically with `max_samples` set to the size of the training set. Instead of building a `BaggingClassifier` and passing it a `DecisionTreeClassifier`, you can instead use the `RandomForestClassifier` class, which is more convenients and optimized for Decision Trees[^21] (similarly, there is a  `RandomForestRegressor` class for regression tasks).
>
> ```python
> from sklearn.ensemble import RandomForestClassifier
> 
> rf_clf = RandomForestClassifier(n_estimators=500, 										max_leaf_nodes=16, 										n_jobs=-1)
> rf_clf.fit(X_train, y_train)
> y_pred = rf_clf.predict(X_test)
> ```
>
> With a few exceptions, a `RandomForestClassifier` has all the hypeparameters of a `DecisionTreeClassifier` (to control how trees are grown), plus all the hypeparameters of a `BaggingClassifier` to control the ensemble itself.
>
> **2. Extra-Trees** 
>
> When you are growing a tree in a Random Forest, at each node only a random subset of the features (the $k$ set) is considered for splitting. It is possible to make trees even more random by also using random thresholds (the $t_k$ value) for each feature rather than searching for the best possible thresholds (like regular Decision Trees do).
>
> A forest with such extremely random trees is called an *Extremely Randomized Trees* ensemble (or *Extra-Trees* for short). Once again, this technique trades more bias for a lower variance. It also makes *Extra-Trees* much faster to train than regular Random Forests[^22].
>
> It is hard to tell in advance whether a `RandomForestClassifier` will preform better or worse than an  `ExtraTreesClassifier` . Generally, the only way to know is to try both and compare them using cross-validation (tuning the hyperparameters uisng grid search).



### BOOSTING

> Boosting (original called *hypothesis boosting*) refers to any Ensemble method that can combine several weak learners into a strong learner. The general idea most boosting methods is to train predictors sequentially, each trying to correct its predecessor. The most popular boosting methods by far are
>
> - AdaBoost (short for Adaptive Boosting) and
> - Gradient Boosting.
>
> **1. AdaBoost**
>
> One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that the predecessor underfitted. This results in new predictors focusing more and more on the hard case. This technique used by AdaBoost.
>
> For example, when training an AdaBoost classifier, the algorithm first train a base classifier (such as a Decision Tree) and uses it to make predictions on the train set. The algorithm then increase the relative weight of misclassified training instances. Then it trains a second classifier, using the updated weights, and again makes predictions on the training set, updates the instance weights, and so on (see Figure 7-7).
>
> ![adaboost](./images/hands-onML_ensemble_adaboost.png)
>
> Once all predictors are trained, the ensemble makes predictions very much like bagging or pasting, except that predictors have different weights depending on their overall accuracy on their corresponding weighted training set.
>
> There is one important drawback to this sequential learning technique: it cannot be parallelized (or only partially), since each predictor can only be trained after the previous predictor has ben trained and evaluated. As a result, it does not scale as well as bagging or pasting.
>
> 
>
> **2. Gradient Boosting** 
>
> Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor. However, instead of tweaking the instance weights at every iteration like AdaBoost does, this method tries to fit the new predictor to the *residual errors* made by the previous predictor.
>
> ```python
> # Gradient Tree Boosting for regression task,
> # a.k.a., Gradient Boosted Regression Trees (GBRT)
> from sklearn.tree import DecisionTreeRegressor
> 
> dtree_reg1 = DecisionTreeRegressor(max_depth=2)
> dtree_reg1.fit(X, y)
> # train a second regressor on the residual errors made by the previous predictor
> y2 = y - dtree_reg1.predict(X)
> dtree_reg2 = DecisionTreeRegressor(max_depth=2)
> dtree_reg2.fit(X, y2)
> # train a third regressor on the residual errors made by the previous predictor
> y3 = y2 - dtree_reg2.predict(X)
> dtree_reg3 = DecisionTreeRegressor(max_depth=2)
> dtree_reg3.fit(X, y3)
> # ensemble contains three trees which makes predictions on a new instance simply by adding up the predictions of all the trees
> y_pred = sum(tree.predict(X_new) for tree in (dtree_reg1, dtree_reg2, dtree_reg3))
> ```
>
> A simpler way to train GBRT ensembles is to use sklearn `GradientBoostingRegressor` class. Much like the `RandomForestRegressor` class, it has hyperparameters to control the growth of Decision Trees.
>
> ```python
> from sklearn.ensemble import GradientBoostingRegressor
> 
> gbrt = GradientBoostingRegressor(max_depth=2,
>                                  n_estimators=3,
>                                  learning_rate=1.0
>                                 )
> gbrt.fit(X, y)
> ```
>
> 
>
> ![gbrt model fitting](./images/hands-onML_ensemble_gradient_boosting.png)
>
> The `learning_rate` hyperparameter scales the contribution of each tree. 
>
> If you set it to a low value, such as 0.1, you will need more trees in the ensemble to fit the training set, but the predictions will usually better. This is a regularization technique called **shrinkage**. In order to find the optimal number of trees, you can use early stopping which can simply implemented by setting`warm_start=True` :
>
> ```python
> import numpy as np
> from sklearn.model_selection import train_test_split
> from sklearn.metrics import mean_square_error
> 
> X_train, y_train, X_test, y_test = train_test_split(X, y)
> gbrt = GradientBoostingRegressor(max_depth=2,
>                                  warm_start=True,
>                                 )
> def early_stop_gbrt(model, n_estimators=200, n_rounds=5,
>                     train_data=(X_train, y_train),
>                     val_data=(X_test, y_test),
>                    ):
>     min_val_error = float("inf")
>     error_going_up = 0
>     for n in range(1, n_estimators):
>         model.n_estimators = n
>         model.fit(train_data)
>         y_pred = model.predict(val_data[0])
>         val_error = mean_square_error(val_data[1], y_pred)
>         if val_error < min_val_error:
>             min_val_error = val_error
>             error_going_up = 0
>         else:
>             error_going_up += 1
>             if error_going_up == n_round:
>                 break
>     return model
> 
> gbrt = early_stop_gbrt(gbrt)
> ```
>
> Note that there is an optimized implementation of Gradient Boosting out there called "XGBoost" which stands for Extreme Gradient Boosting,  it is a popular Python library aimed to be extremely fast, scalable, and portable.



[^17]: 集成学习(ensemble learning，a.k.a, multi-classifier system, committee-based learning)通过构建并结合多个学习器来完成学习任务。集成学习一般结构是，先产生一组 “个体学习器(individual learner)”，再用某种策略将它们结合起来。个体学习器通常由一个现有学习算法从训练数据产生，例如 C4.5决策树算法或BP神经网络算法等，此时，如果集成中只包含同种类型的个体学习器，如 “决策树集成”、“神经网络集成”等，则这样的集成是 “同质” 的(homogeneous)集成，同质集成中的个体学习器也称为 “基学习器(base learner)”，相应的学习算法称为 “基学习算法(base learning algorithm)”；反之，则是 “异质” 的(heterogenous)集成，这时个体学习器常称为 "组件学习器(component learner)" 或直接称为个体学习器。
[^18]: Boosting 是一族可将弱学习器提升为强学习器的算法。这族算法的工作机制类似：先从初始训练集训练出一个机学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直至学习器数目达到事先指定的值$T$，最终将这$T$个基学习器进行加权结合。

[^19]: bootstrapping 采样是有放回的随机重复采样，样本在$m$次采样中始终不被采到的概率是$(1 - {1 \over m})^m$，取极限得到$lim_{(m \rightarrow \infin)} (1 - {1 \over m})^m \rightarrow {1 \over e} \approx 0.368$，即通过自助采样，初始数据集中约有36.8%的样本未出现在自助采样集中。自助法能从初始数据集中产生多个不同训练集，这对集成学习有很大好处。然而，自助法产生的数据集改变了初始数据集的分布而引入估计偏差，因此，在初始数据量足够时，留出法和交叉验证法更常用一些。
[^20]: statistical mode: 众数。也就是频率最高的预测类别，与 hard voting classfifer 类似。
[^21]: The `BaggingClassifier` class remains useful if you want a bag of something other than Decision Trees.
[^22]: Finding the best possible threshold for each feature at every node is one of the most time-consuming tasks of growing a tree.



## Part-II: Engineering views

## Sklearn Random-forest model

```python
sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, 					criterion='gini', max_depth=None, min_samples_split=2, 					min_samples_leaf=1, min_weight_fraction_leaf=0.0, 						max_features='auto', max_leaf_nodes=None, 								min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, 		oob_score=False, n_jobs=None, random_state=None, verbose=0, 			warm_start=False, class_weight=None, ccp_alpha=0.0, 					max_samples=None)
```

A random forest classifier.

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the `max_samples` parameter if `bootstrap=True` (default), otherwise the whole dataset is used to build each tree.

### feature_importances_

Sklearn measures a feature's importance by looking at how much the tree nodes that use that feature reduce impurity on average (across all trees in the forest). More precisely, it is a weighted average, where each node's weight is equal to the number of training samples that are associated with it.

`feature_importances_` is a impurity-based feature importances.

The higher, the more importance the feature. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.

Warning: impurity-based feature importance can be misleading for high cardinality features (many unique values). see [permutation_importance](###permutation importance) as an alternative.

Tree-based models measure the feature importances based on the [mean decrease in impurity](#MDI). Impurity is quantified by the splitting criterion of the decision trees (Gini, Entropy(i.e., imformation gain) or Mean Square Error). However, this method can give high importance to features that may not be predictive on unseen data when the model is overfitting. Permutation-based feature importance, on the other hand, avoids this issue, since it can be computed on unseen data (hold-out set, validation set, etc).

Furthermore, impurity-based feature importance for trees are **strongly biased** and **favor high cardinality features** (typically numerical features) over low cardinality features such as binary features or categorical variables with a small number of possible categories. [(see this explanation)](####增益率).

The following example highlights the limitations of impurity-based feature importance in contrast to permutation-based feature importance:

[Permutation importance vs Random Forest Feature Importance (MDI)](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py)

### permutation importance

```python
sklearn.inspection.permutation_importance(estimator, X, y, *,
                                          scoring=None,
                                          n_repeats=5,
                                          n_jobs=None,
                                          random_state=None
                                         )
```

Permutation importance for feature evaluation.

The `estimator` is required to be a fitted estimator. `X` can be the data set used to train the estimator or a hold-out set. The permutation importance of a feature is calculated as follows.

First, a baseline metric, defined by [scoring](#Scoring Parameter), is evaluated on a (potentially different) dataset defined by `X`. Next, a feature column from the validation set is permuted and the metric is evaluated again. 

The permutation importance is defined to be difference between the baseline metric and metric from permutating the feature column.

---

Algorithm 1. Permutation importance

---

Inputs: fitted predictive model $m$, tabular dataset (training or validation) $D$.

Compute the reference score $s$ of the model $m$ on data $D$ (for instance the accuracy for a classifier or the $R^2$ for a regressor).

**For** each feature $j$ (column of $D$):

​	**For** each repetition $k$ in $1, \ldots, K$:

​		Randomly shuffle column $j$ of dataset $D$ to generate a corrupted version of the data named $\tilde{D}_{k,j}$.

​		Compute the score $s_{k,j}$ of model $m$ on corrupted data $\tilde{D}_{k,j}$.

​	Compute importance $i_j$ for feature $f_j$ defined as:
$$
i_j = s - {1 \over K} \sum^K_{k=1}s_{k,j}.
$$

---

### Misleading values on strongly correlated features

When two features are correlated and one of the feature is permuted, the model will still have access to the feature through its correlated feature. This will result in a lower importance value for both features, where they might actually be important.

One way to handle this is to cluster features that are correlated and only keep one feature from each cluster. This strategy is explored in the following example:

[Permutation Importance with Multicollinear or Correlated Features](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py).



### Does modeling with Random Forests require cross-validation?

<p style="text-align:center;color:blue;">
    "Random forests provide free cross-validation."
</p>

The `RandomForestClassifier` is trained using *bootstrap aggregation[^1]*, where each new tree is fit from a bootstrap sample of the training observations $z_i = (x_i, y_i)$. The out-of-bag (OOB) error is the average error for each $z_i$ calculated using predictions from the tress that do not contain $z_i$ in their respective bootstrap sample. This allows the `RandomForestClassifier` to be fit and validated whilst being trained.

> By principle since it randomizes the variable selection during each tree split, it's not prone to overfit unlike other models. However if you want to use CV using nfolds in sklearn you can still use the concept of hold-out set such as `oob_score=True` which shows model performance with or without using CV.



### Plot the decision tree













