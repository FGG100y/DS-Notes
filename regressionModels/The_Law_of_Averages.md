
```html
**Important Note**:

Almost all the contents (text, images) are came from these great books and
online resources:

* Statistics, by David Freeman, Robert Pisani, and Roger Perves
```

## The Law of Average

With a large number of tosses (a coin), the size of the difference between the number of heads and the expected number is likely to be quite large in absolute terms. But compared to the number of tosses, the difference is likely to be quite small. That's the law of average.

![coin-tossing1](./images/stats_the_law_of_averages_1.png)

Q: In 10000 tosses one would expect to get 5000 heads, right?

A: Right. 

A: But not exactly. One only expect to get around 5000 heads, means that one could get 5001 or 4998 or 5007. The amount off 5000 is what we call "chance error".

To be more specific, here is the equation:
$$
\text{number of heads = half the number of tosses + chance error}
$$
This error is likely to be large in absolute terms, but small compared to the number of tosses.

What **the law of averages** says is that the number of heads will be around half the number of tosses, but it will be off by some amount -- chance error. As the number of tosses goes up, the chance error gets bigger in absolute terms. Compared to the number of tosses, it get smaller.

![coin-tossing1](./images/stats_the_law_of_averages_2.png)

<p style="text-align: center;">Figure 2 shows the law of averages</p>

Q: How big the chance error is likely to be?

A: Well, with 100 tosses, the chance error is likely to be around 5 in the size. With 10000 tosses, the chance error is likely to be around 50 in size. Multiplying the number of tosses by 100 only multiplies the likely size of the chance error by $\sqrt{100} = 10$.



### Chance Processes

To what extent are the numbers influenced by chance? This sort of question must be faced over and over again in statistics. A general strategy will be presented in the next few chapters. The two main ideas:

- Find an **analogy** between the process being studied (sampling voters in the poll example) and **drawing numbers at random from a box**.
- Connect the variability you want to know about (for example, in the estimate for the Democratic vote) with the chance variability in the sum of the numbers drawn from the box.

The analogy between a chance process and drawing from a box is called a **box model**. The point is that the chance variability in the sum of the numbers drawn from a box will be easy to analyze mathematically. More complicated processes can then be dealt with through the analogy.



### The Sum of Draws

At first, this example may seen artificial. But it is just like a turn at Monopoly -- you roll a pair of dice, add up the two numbers, and move that many squares. Rolling a die is just like picking a number from the box.

![box model 1](./images/stats_box_model_1.png)

Next, imagine taking 25 draws from the same box (draw with replacement).

About how big is their sum going to be? The most direct way to find out is by experiment. The Python code block below can mimic this:

```python
import random

def draw_numbers(box=None, k=1):
    """Simulation of drawing numbers from a box
    
    :box: contains numbers to be draw
    :k: how many numbers to draw at a time (default 1)
    """
    start, end = 1, 6
    box = range(start, end) if not box else box
    return random.choices(box, k)

def sum_draws(box, k, n=25, debug=False):
    """Sum the number of n draws"""
    draws = [draw_numbers(box, k) for i in range(n)]
    try:
        sum_of_draws = sum(draws)
    except Exception:
        # k > 1
        sum_of_draws = sum([sum(b) for b in draws])
    if debug:
        print(draws)
    return sum_of_draws
```

The *sum of draws* from a box is shorthand for the process discussed in previous:

- Draw tickets at random from a box.
- Add up the numbers on the tickets.



## The Expected Value and Standard Error

A chance process is running. It delivers a number. Then another. And another. You are about to drown in random output. But mathematicians have found a little order in this chaos. Then numbers delivered by the process vary around the **expected value**, the amounts off being similar in size to the **standard error**.

### The expected value

To be more specific: count the number of heads in 100 tosses of a coin. You might get 57 heads. This is 7 above the expected value of $50$, so the *chance error* is $+7$. Made another 100 tosses, got 46 heads, so the *chance error* would be $-4$. And so on and so forth. Your numbers will be off $50$ (the expected value) by chance amounts similar in size to the **standard error**, which is $5$.

The formulas for the expected value and standard error depend on the chance process which generates the number. This chapter deals with the sum of draws from a box, and the formula for the expected value will be introduced with an example: the sum of 100 draws made at random with replacement from the box

```python
box = [1, 1, 1, 5]
```

About how large should this sum be? To answer this question, think how the draws should turn out. There are four tickets in the box, so `5` should come up on around one-fourth of the draws, and `1` on three-fourths. With 100 draws, you can expect to get around twenty-five `5`'s and seventy-five `1`'s. The sum of the draws should be around
$$
25 \times 5 + 75 \times 1 = 200.
$$
That is the expected value.

The formula for the expected value is a short-cut. It has two ingredients:

- the number of draws;
- the average of the numbers in the box, abbreviated to "average of box."

<div>
    <p style="text-align: justify;color:#CE5937;">The expected value for the sum of draws made at random with replcement from a box equals
    </p>
    <p style="text-align: center;color:#CE5937;"> 
        (number of draws) &times; (average of box).
    </p>
</div>


To see the logic behind formula, go back to the example. The average of the box is
$$
{{1 + 1 + 1 + 5} \over 4} = 2
$$
On the average, each draw adds around 2 to the sum. With 100 draws, the sum must be around $100 \times 2 = 200$.

### The Standard Error

Suppose 25 draws are made at random with replacement from the box

```python
box = [0, 2, 3, 4, 6]
```

Each of the five tickets should appear on about one-fifth of the draws, that is 5 times. So the sum should be around
$$
5 \times 0 + 5 \times 2 + 5 \times 3 + 5 \times 4 + 5 \times 6 = 75.
$$
That is the expected value for the sum. Of course, each ticket won't appear on exactly one-fifth of the draws, just as in the heads of coin tosses. The sum will be off the expected value by a chance error:
$$
\text{sum} = \text{expected value} + \text{chance error}.
$$
The $\text{chance error}$ is the amount above (+) or below (-) the expected value. For example, if the sum is 70, the chance error is -5.

How big the chance error likely to be? The answer is given by the **standard error**, usually abbreviated to **SE**.

<p style="text-align: justify;color:#CE5937;">A sum is likely to be around its expected value, but to be off by a chance error similar in size to the standard error.</p>

There is a formula to use in computing the SE for a sum of draws made at random with replacement from a box.  It is called the **square root law**:

<div>
    <p style="text-align: justify;color:#CE5937;">The square root law. When drawing at random with replacement from a box of numbered tickets, the standard error for the sum of the draws is
    </p>
</div>


$$
\sqrt{\text{number of draws}}\ \times \ (\text{SD of box}).
$$

> Recall that the SD measures the spread among the numbers in the box.
>
> The SD says how far away numbers on a list are from their average. Most entries on the list will be somewhere around one SD away from the average. Very few will be more than two or three SDs away.

The SD and SE are different. The SD applies to spread in lists of numbers. It is worked out using the method explained on p. 71. By contrast, the SE applies to chance variability -- for instance, in the sum of the draws.

![SD_n_SE](./images/stats_SD_and_SE_applications.png)

At the beginning of the section, we looked at the sum of 25 draws made at random with replacement from the box

```python
box = [0, 2, 3, 4, 6]
```

The expected value for this sum is 75. The sum will be around 75, but will be off by a chance error. How big is the chance error likely to be? To find out, calculate the standard error.
$$
\text{The average of number of the box: } 3  \\
\text{The deviation from the average are: }{-3\ -1\ 0\ 1\ 3} \\
\text{The SD of the box is: }
\sqrt{\frac{(-3)^2 + (-1)^2 + 0^2 + 1^2 + 3^2}{5}} = \sqrt{{20 \over 5}} = 2.
$$
This measures the variability in the box. According to the square root law, the sum of 25 draws is more variable, by the factor $\sqrt{25} = 5$. The SE for the sum of 25 draws is $5 \times 2 = 10$. In other words, the likely size of the chance error is $10$. And the sum of the draws should be around 75, give or take 10 or so. 

**In general, the sum is likely to be around its expected value, give or take a standard error or so.**

### Using the Normal Curve

A large number of draws will be made at random with replacement from the box. What is the chance that the sum of the draws will be in a given range? Mathematician discovered the normal curve while trying to solve problems of this kind.

Suppose the computer is programmed (see the Python code block in previous section) to take the sum of 25 draws made at random with replacement from the magic box

```python
box = [0, 2, 3, 4, 6]
```

It prints out the result, repeating the process over and over again. About what percentage of the observed values should be between 50 and 100?

Each sum will be somewhere on the horizontal axis between $0$ and $25 \times 6 = 150$.

To find the chance, convert to standard units and use the normal curve. Standard units say how many SEs a number is away from the expected value. In the example, 100 becomes 2.5 in standard units ($\text{(entry - EV)/SE} = (100 - 75) / 10 = 2.5$); similarly, 50 becomes $-2.5$. 

![SE_n_SD_2](./images/stats_SD_and_SE_applications2.png)

The interval from 50 to 100 is the interval within 2.5 SEs of the expected value, so the sum should be there about 99% of the time.



### A Short-Cut

Finding SDs can be painful, but there is a short-cut for lists only two different numbers, a big one and a small one. (Each number can be repeated several times.)

When a list has only two different numbers ("big" and "small"), the SD equal
$$
(\text{big} - \text{small}) \times \sqrt{\text{(fraction with big)} \times \text{(fraction with small)}}
$$
For example, take the list [5, 1, 1, 1]. The short-cut can be used, the SD is
$$
(5 - 1) \times \sqrt{{1 \over 4} \times {3 \over 4}} \approx 1.73
$$
The short-cut involves much less arithmetic than finding the root-mean-square of the deviations from the average, and gives exactly the same answer.

**It is time to connect the square root law and the law of averages.** Suppose a coin is tossed a large number of times. Then heads will come up on about half the tosses:
$$
\text{number of heads} = \text{half the number of tosses} + \text{chance error.}
$$
How big is the chance error likely to be?

According to the square root law, the likely size of the chance error is
$$
\begin{eqnarray}
\sqrt{\text{number of tosses}} \times SD 
&=& \sqrt{\text{number of tosses}} \times \bigg((1 - 0) \times \sqrt{{1 \over 2} \times {1 \over 2}} \bigg) \\
&=& \sqrt{\text{number of tosses}} \times {1 \over 2}.
\end{eqnarray}
$$
For instance, with 10000 tosses the standard error is $\sqrt{10000} \times {1 \over 2} = 50$. When the number of tosses goes up to 1000000 the standard error goes up too, but only to 500 -- because of the square root. As the number of tosses goes up, the SE for the number of heads gets bigger and bigger in absolute terms, but smaller and smaller relative to the number of tosses. That is why the percentage of heads gets closer and closer to $50\%$. **The square root law is the mathematical explanation for the law of averages.**

