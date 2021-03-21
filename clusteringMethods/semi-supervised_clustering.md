## 半监督聚类 （semi-supervised clustering）

聚类是一种典型的无监督学习任务，然而在现实聚类任务中我们往往能获得一些额外的监督信息，于是可以通过半监督聚类来利用额外监督信息以获得更好的聚类效果。

聚类任务中获得额外监督信息大致有两种类型：

- 样本约束：

  必连 (must-link): 指的是样本必属于同一个簇

  勿连 (cannot-link): 样本必不属于同一个簇

- 样本标签：

  监督信息来自少量带有标签的样本



***

### 约束$k$均值算法 (pseudo-code)

约束$k$均值算法 (Constrained k-means) 是利用第一类监督信息的代表。给定样本集 $D=\{x_1, x_2, \ldots, x_m\}$ 以及 “必连” 关系集合 $\cal{M}$ 和 “勿连” 关系集合 $\cal{C}$ ，$(x_i, x_j) \in \cal{M}$ 表示 $x_i, x_j$ 必属于同簇，$(x_i, x_j) \in \cal{C}$ 表示 $x_i, x_j$ 必不属于同簇。该算法是 $k$-means 算法的扩展，它在聚类过程中要确保样本的约束得到满足，否则返回错误提示，算法如下：

***

**输入**： 样本集 $D = \{x_1, x_2, \ldots, x_m\}$;

​			必连约束集合 $\cal{M}$ ;

​			勿连约束集合 $\cal{C}$ ;

​			聚类簇数 $k$.

过程：

01: 从 $D$ 中随机选取 $k$ 个样本作为初始均值向量 $\{\mu_1, \mu_2, \ldots, \mu_k\}$;

02: **repeat**

03:	 $C_j = \phi (1 \le j \le k)$;

04:	 **for** $i = 1, 2, \ldots, m$ **do**

05:	 	计算样本 $x_i$ 与各个均值向量 $\mu_j (1 \le j \le k)$ 的距离： $d_{ij} = ||x_i - \mu_j||_2$;

06: 		$\cal{K} = \{1, 2, \ldots, k\}$;

07: 		$\text{is_merged} = false$;

08: 		**while** $\neg\ \text{is_merged}$ **do**

09: 			基于 $\cal{K}$ 找出与样本 $x_i$ 距离最近的簇： $r = argmin_{j \in \cal{K}}\ d_{ij}$;

10: 			检测将 $x_i$ 划入聚类簇 $C_r$ 是否会违背 $\cal{K}$ 与 $\cal{C}$ 中的约束；

11：			**if**  $\neg\ \text{is_voilated}$  **then**

12: 				$C_r = C_r \cup \{x_i\}$;

13: 				$\text{is_merged} = true$

14: 			**else**

15: 				$\cal{K} = K \setminus \{r\}$;

16: 				**if** $\cal{K} = \phi$ **then**

17: 					**break** 并返回错误提示

18: 				**end if**

19: 			**end if**

20: 		**end while**

21: 	**end for**

22: 	**for** $j = 1, 2, \ldots, k$ **do**

23: 		$\mu_j = {1 \over |C_j|} \sum_{x \in C_j} x$;

24: 	**end for**

25: **until** 均值向量均为更新

**输出**：簇划分 $\{C_1, C_2, \ldots, C_k\}$

***



---

### 约束种子 $k$ 均值算法 (pseudo-code)

约束种子 $k$ 均值算法 (Constrained Seed k-means) 利用第二种监督信息，即少量有标记样本（此处样本标记指的是簇标记‘cluster label’，而不是类别标记‘class label’）。给定样本集 $D = \{x_1, x_2, \ldots, x_m\}$，假定少量的有标记样本为 $S = \cup^k_{j=1} S_j \subset D$，其中，$S_j \ne \phi$ 为隶属于第 $j$ 个聚类簇的样本。这样的监督信息利用起来很容易：直接将他们作为‘种子’，用他们初始化 $k$ 均值算法的 $k$ 个聚类中心，并且在聚类簇迭代更新过程中不改变种子样本的簇隶属关系。其算法描述如下：

---

**输入**:	样本集 $D = \{x_1, x_2, \ldots, x_m\}$;

​			少量有标记样本 $S = \cup^k_{j=1} S_j$;

​			聚类簇数 $k$.

过程:

01: **for** $j = 1, 2, \ldots, k$ **do**

02: 	$\mu_j = {1 \over |S_j|} \sum_{x \in S} x$

03: **end for**

04: **repeat**

05: 	$C_j = \phi (1 \le j \le k)$;

06: 	**for** $j = 1, 2, \ldots, k$ **do**

07: 		**for all** $x \in S_j$ **do**

08: 			$C_j = C_j \cup \{x\}$

09: 		**end for**

10: 	**end for**

11: 	**for all** $x_i \in D \setminus S$ **do**

12: 		计算样本 $x_i$ 与各个均值向量 $\mu_j (1 \le j \le k)$ 的距离： $d_{ij} = ||x_i -\mu_j||_2$;

13: 		找出与样本 $x_i$ 距离最近的簇： $r = argmin_{j \in \{1, 2, \ldots, k\}} d_{ij}$;

14: 		将样本 $x_i$ 划入相应的簇： $C_r = C_r \cup \{x_i\}$

15: 	**end for**

16: 	**for** $j = 1, 2, \ldots, k$ **do**

17: 		$\mu_j = {1 \over |C_j|} \sum_{x \in C_j} x$;

18: 	**end for**

19: **until** 均值向量均未更新

**输出**: 簇划分 $\{C_1, C_2, \ldots, C_k\}$

---



