## The Normal Approximation for Probability Histograms

> Everybody believes in the "normal approximation", the experimenters because they think it is a mathematical theorem, the mathematicians because they think it is an experimental fact.
>
> <p align=right>-- G. Lippmann (Franch physicist, 1845-1921)</p>

### Probability Histograms

When a chance process generates a number, the expected value and standard error are a guide to where that number will be. But the probability histogram gives a complete picture.

<p style="text-align: justify;color:#CE5937";>A probability histogram is a new kind of graph. This graph represents chance, not data. A probability histogram represents chance by area.
</p>

Figure 1. The computer simulated rolling a pair of dice and finding the total number of spots (Adding the numbers). The ideal or probability histogram for the total number of spots when a pair of dice are rolled (Converged from empirical histograms).

![probability histogram](./images/stats_probability_histogram.png)

There are 6 chances in 36 of rolling a 7, that's $16 {2 \over 3} \%$. Consequently, the area of the rectangle over 7 in the probability histogram equals $16 {2 \over 3} \%$. Similarly for the other rectangles.

The base of the rectangle is centered at a possible value for the sum of the draws, and the area of the rectangle equals the chance of getting that value. The total area of the histogram is $100 \%$.

Figure 2. Probability histogram of the product of the numbers on a pair of dice.

![probability histogram](./images/stats_probability_histogram2.png)

Figure 2 is very different from figure 1: the new histogram has gaps. Since there is no way to get the product 7 (as well as 11, 13, 17 and so on), the chance is zero. All the gaps can be explained in this way.



### The Normal Approximation

The normal curve has already been used in previous section (in "Using the Normal Curve" section) to figure chances. This section will explain the logic.

![probability histogram](./images/stats_probability_histogram3.png)

$\color{Green}{Example}$ 1. A coin will be tossed 100 times. Estimate the chance of getting:

​	(a) exactly 50 heads.

​	(b) between 45 and 55 heads inclusive. 

​	(c) between 45 and 55 heads exclusive. 

*Solution*. The expected number of the heads is $50$ and the standard error is $5$.

Part (a):

![nc1](./images/stats_normal_curve_e1.png)

Part (b):

![nc1](./images/stats_normal_curve_e2.png)

Part (c):

![nc1](./images/stats_normal_curve_e3.png)

Often, the problem will only ask for the chance that the number of heads is between 45 and 55, without specifying whether endpoints are included or excluded. Then use the compromise procedure:

![nc1](./images/stats_normal_curve_e4.png)

Keeping track of the endpoints has an official name, "the continuity correction". The correction is worthwhile if the rectangles are big, or if a lot of precision is needed.

The normal approximation consists in replacing the actual probability histogram by the normal curve before computing the areas. This is legitimate when the probability histogram follows the normal curve. Probability histogram are often hard to work out, while areas under the normal curve are easy to look up in the table[^1].

**Conclusion**

If we looked at the sum of the draws from different boxes:

```python
b1 = [0, 1]
b2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # nine 0's
b3 = [1, 2, 3]
b4 = [1, 2, 9]
```

There are plenty more where those came from. But the pattern is always the same. With enough draws, the probability histogram for the *sum* will be close to the normal curve. Mathematicians have a name for this fact. They call it **the central limit theorem**, because it plays a central role in statistical theory.

<p style="text-align: justify;color:#CE5937";>The Central Limit Theorem. When drawing at random with replacement from a box, the probability histogram for the sum will follow the normal curve, even if the contents of the box do not. The histogram must be put into standard units, and the number of draws must be reasonably large.
</p>

When the probability histogram does follow the normal curve, it can be summarized by the **expected value** and **standard error**. For instance, suppose you had to plot such a histogram without any further information. In standard units you can do it, at least to a first approximation:

![probability histogram](./images/stats_probability_histogram4.png)

To finish the picture, you have to translate the standard units back into original units by filling in the question marks. This is what the **expected value** and **standard error** do. They tell you almost all there is to know about this histogram, because it follows the normal curve.

<p style="text-align: justify;color:#CE5937";>The expected value pins the center of the probability histogram to the horizontal axis, and the standard error fixes its spread.
</p>

According to the **square root law**, the expected value and standard error for a sum can be computed from

- the number of draws,
- the average of the box, ($\Rightarrow EV = \text{(number of draws)} \times \text{average}$)
- the SD of the box. ($\Rightarrow SE = \sqrt{ \text{number of draws} }\times \text{SD}$)

These three quantities just about determine the behavior of the sum.

NOTE that it has to be shown that the process generating the data is like drawing numbers from a box and taking the sum.

[^1]:  The normal table (see detail in Section TABLES).
