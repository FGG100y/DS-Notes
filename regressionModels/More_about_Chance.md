
```html
**Important Note**:

Almost all the contents (text, images) are came from these great books and
online resources:

* Statistics, by David Freeman, Robert Pisani, and Roger Perves
```

## More about Chance

### FAQs

- What's the difference between mutually exclusive and independent?
- When do I add and when do I multiply?

"Mutually exclusive" is one idea; independence is another. Both ideas apply to pairs of events, and say something about how the events are related. However, the relationships are quite different.

- Two events are mutually exclusive if the occurrence of one prevents the other from happening.
- Two events are independent if the occurrence of one does not change the chance for the other.

The addition rule, like the multiplication rule, is a way of combining chances. However, the two rules solve different problems.

- The addition rule finds the chance that **at least** one of two things happens.
- The multiplication rule finds the chance that two things **both** happen.

So the first in deciding whether to add or to multiply is to read the question: Do you want to know $P(A\ \text{or}\ B)$, $P(A\ \text{and}\ B)$, or something else entirely? But there is also a second step -- because the rules apply only if the events are related in the right way:

- Adding the probabilities of two events requires them to be mutually exclusive.
- Multiplying the unconditional probabilities of two events requires them to be independent.  (For dependent events, the multiplication rule uses conditional probabilities.)

If the chance of an event is hard to find, try to find the chance of the opposite event. Then subtract from 100%. This is useful when the chance of the opposite event is easier to compute.

### The Binomial Formula

Suppose a chance process is carried out as sequence of trials. An example would be rollling a die 10 times, where each roll counts as a trial. There is an event of interest which may or may not occur at each trial: the die may or may not land ace. The problem is to calculate the chance that event will occur a specified number of times.

---

The chance that an event will occur exactly $k$ times out of $n$ is given by the binomial formula
$$
{n! \over {k!(n-k)!}} p^k (1 - p)^{n-k}
$$
where $n$ is the number of trials, $k$ is the number of times the event is to occur, and $p$ is the probability that the event will occur on any particular trial. The assumptions:

- The value of $n$ must be fixed in advance.
- $p$ must be the same for trial to trial.
- The trials must be independent.

---

The formula starts with the binomial coefficient,
$$
{n! \over {k!(n-k)!}}
$$
Recall that this is the number of ways to arrange $n$ objects in a row, when $k$ are alike of on kind and ($n - k$) are alike of another (for instance, red and green marbles).

> Mathematicians usually write ${n \choose k}$ for the binomial coefficient:
> $$
> {n \choose k} = {n! \over {k!(n-k)!}}
> $$
> They read $n \choose k$ as "n choose k," the idea being that the formula gives the number of ways to choose $k$ things out of $n$. Older books write the binomial coefficient as $_n C_k$ or $^n C_k$, the "number of combinations of $n$ things taken $k$ at a time."

$\color{Green}{Example}$ 1. A die is rolled 10 times. What is the chance of getting exactly 2 aces?

*Solution*. 

​	The number of trials is fixed in advance. It is 10. So $n = 10$.

​	The event of interest is rolling an ace. The probability of rolling an ace is the same from trial to trial. It is $1 \over 6$. So $p = {1 \over 6}$.

​	The trials are independent. All together, the binomial formula can be used, and the answer is
$$
{10! \over 2!8!} \bigg({1 \over 6} \bigg)^2 \bigg({5 \over 6} \bigg)^8 \approx 29\%
$$
​	
