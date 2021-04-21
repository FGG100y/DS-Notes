## Pseudo-code

### Similarity (Distance) Measure

- Euclidian distance ($L_2$ norm):
  $$
  L_2(x, x') = \sqrt{\sum^{N}_{i=1}(x_i - x'{_i})^2}
  $$

- $L_1$ norm:
  $$
  L_2(x, x') = \sqrt{\sum^{N}_{i=1}|x_i - x'{_i}|}
  $$

- Cosine similarity:
  $$
  \cos(x, x') = \frac{x*x'}{||x||\ ||x'||}
  $$

- Kernels



### Clustering Criterion

- Evaluation function that assigns a (usually real-valued) value to a clustering
  - Clustering criterion typically function of
    - within-cluster similarity, and
    - between-clustering dissimilarity
- Optimization
  - Find clustering that maximizes the criterion
    - Global optimization (often intractable)
    - Greedy search (K-Means)
    - Approximation algorithms (GMMs)



### Centroid-based Clustering

- Assumes instances are real-valued vectors

- Clusters represented via *centroids* (i.e., average of points in a cluster) **c**:
  $$
  \mu(c) = {1 \over |c|} \sum_{x \in c} x
  $$

- Reassignment of instances to clusters is based on *distance* to the current cluster centroids



### K-Means 

---

K-Means({$x_1, x_2, \ldots, x_N$}, K)

​	1	($s_1, s_2, \ldots, s_K$) $\leftarrow$ SelectRandomSeeds({$x_1, x_2, \ldots, x_N$}, K)  # (vanilla: select K instances)

​	2	**for** k $\leftarrow$ 1 to K

​	3	**do** $\mu_k \leftarrow s_k$ 

​	4	**while** stopping criterion has not been met

​	5	**do for** k $\leftarrow$ 1 to K

​	6		**do** $C_k \leftarrow$ {}

​	7		**for** n $\leftarrow$ 1 to N

​	8		**do** $j \leftarrow \text{argmin}_{j'}|\mu_{j'} - x_n|$ 

​	9			$w_j \leftarrow w_j \cup \{x_n\}$  # (reassignment of vectors)

  10		**for** k $\leftarrow$ 1 to K

  11		**do** $\mu_k \leftarrow {1 \over |C_k|} \sum_{x \in C_k} x$  # (recomputation of centroids)

  12	**return** {$\mu_1, \mu_2, \ldots, \mu_K$}

---

