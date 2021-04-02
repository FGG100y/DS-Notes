# Sampling

## Sample Surveys

### Simple Random Sampling

---

<p style="text-align: justify;color:#CE5937;">
    Simple random sampling means drawing at random without replacement.
</p>


---

Most of the times, it just is not practical to take a simple random sample. Consequently, most survey organizations use a probability method called *multistage cluster sampling*.

### Multistage cluster sampling

The Gallup Poll makes a separate study in each of the four geographical regions of the United States—Northeast, South, Midwest, and West (figure 1). One such grouping might be all towns in the Northeast with a population between 50 and 250 thousand. Then, a random sample of these towns is selected. Interviewers are stationed in the selected towns, and no interviews are conducted in the other towns of that group. Other groupings are handled the same way. This completes the first stage of sampling.

For election purposes, each town is divided up into *wards*, and the wards are subdivided into *precincts*. At the second stage of sampling, some wards are selected -- at random -- from each town chosen in the stage before. At the third stage, some precincts are drawn at random from each of the previously selected wards. At the fourth stage, households are drawn at random from each selected precinct. Finally, some members of the selected households are interviewed. Even here, no discretion is allowed. For instance, Gallup Poll interviewers are instructed to “speak to the youngest man 18 or older at home, or if no man is at home, the oldest woman 18 or older.”

![mscs](./images/stats_sampling_mscs.png)

This design offers many of the advantages of quota sampling. Each stage in the selection procedure uses an objective and impartial chance mechanism to select the sample units. This completely eliminates the worst feature of quota sampling: selection bias on the part of the interviewer.

Simple random sampling is the basic probability method. Other methods can be quite complicated. But all probability methods for sampling have two important features:

- the interviewers have no discretion at all as to whom they interview;
- there is a definite procedure for selecting the sample, and it involves the planned use of chance.

As a result, with a probability method it is possible to compute the chance that any particular individuals in the population will get into the sample.

### How Well Do Probability Methods Work?

![Gallup Poll predictions](./images/stats_gallup_poll_table.png)

Why do probability methods work so well? At first, it may seem that judgment is needed to choose the sample. For instance, quota sampling guarantees that the percentage of men in the sample will be equal to the percentage of men in the population. With probability sampling, we can only say that the percentage of men in the sample is likely to be closed to the percentage in the population: certainty is reduced to likelihood.

But judgment and choice usually show bias, while chance is impartial. That is why probability methods work better than judgment.

**The 1992 election.** In 1992, there was a fairly large percentage of undecided respondents, and Gallup allocated all of them to Clinton. That turned out to be a bad idea. Many of the undecided seem in the end to have voted for Perot, explaining Gallup’s large error for the 1992 election (table 4). Predicted and actual votes for Clinton, Bush, and Perot are shown below.

![Gallup Poll predictions](./images/stats_gallup_poll_table2.png)



### Chance Error and Bias

To focus on the issue, imagine a box with a very large number of tickets, some marked $1$ and the others marked $0$. That is the **population**. A survey organization is hired to estimate the percentage of $1$'s in the box. That is the **parameter**. The organization draws 1000 tickets at random without replacement. That is the **sample**.

The estimate is still likely to be a bit off, because the sample is only part of the population. Since the sample is chosen at random, the amount off is governed by chance:
$$
\text{percentage of 1's in sample} = \text{percentage of 1's in box} + \text{chance error}.
$$
Now there are some questions to ask about chance errors:

- How big are they likely to be?
- How do they depend on the size of the sample? the size of the population?
- How big does the sample have to be in order to keep the chance errors under control?

In more complicated situations, the equation has to take $\text{bias}$ into account:
$$
\text{estimate} = \text{parameter} + \text{bias} + \text{chance error}.
$$
Chance error is often called "sampling error": the "error" comes from the fact that the sample is only part of the whole. Similarly, bias is called "non-sampling error" -- the error from other sources, like non-response.

Bias is often more serious problem than chance error, but methods for assessing bias are not well developed. Usually, "bias" means prejudice. For a statistician, bias just means any kind of systematic error in an estimate.

### SUMMARY

1. A *parameter* is a numerical fact about a population. Usually a parameter cannot be determined exactly, but can only be estimated.
2. A *statistic* can be computed from a sample, and used to estimate a parameter. A statistic is what the investigator knows. A parameter is what the investigator wants to know.
3. Some methods for choosing samples are likely to produce accurate estimates. Others are spoiled by *selection bias* or *non-response bias*. When thinking about a sample survey, ask yourself:
   - What is the population? the parameter?
   - How was the sample chosen?
   - What was the response rate?

4. Large samples offer no protection against bias (the sampling method matters).
5. Even when using probability methods, bias may come in. Then the estimate differs from the parameter, due to bias and chance error.



## Chance Errors in Sampling

### The Expected Value and Standard Error

---

<p style="text-align: justify;color:#CE5937;">
    With a simple random sample, the expected value for the sample percentage equals the population percentage.
</p>


---

However, the sample percentage will not be exactly equal to its expected value -- it will be off by a chance error. How big is this error likely to be? The answer is given by the **standard error**.

To compute an SE, you need a box model. The sociologist took a sample of size 100 from a population consisting of 3091 men and 3581 women. She classified the people in the sample by sex and count the men. So there should be only $1$'s and $0$'s in the box. The number of men in the sample is like the sum of 100 draws from the box

```python
box = [1] * 3091 + [0] * 3581.
```

She used a simple random sample, so the tickets must be drawn without replacement. This completes the box model.

The fraction of $1$'s in the box is 0.46. Therefore, the SD of the box is $\sqrt{0.46 \times 0.54} \approx 0.50$. The SE for the sum of 100 draws is $\sqrt{100} \times 0.5 = 5$. The sum of 100 draws from the box will be around 46, give or take 5 or so.

Now 46 out of 100 is $46 \%$ and 5 out of 100 is $5 \%$. Therefore, the percentage of men in the sample is likely to be around 46%, give or take 5% or so. This $5 \%$ is the SE for the percentage of men in the sample.

---

<p style="text-align: justify;color:#CE5937;">
    To compute the SE for a percentage, first get the SE for the corresponding number; then convert to percent, relative to the size of the sample. As a cold mathematical formula,
</p>


$$
\text{SE for percentage} = {\text{SE for number} \over \text{size of sample}} \times 100 \%.
$$

<p style="text-align: justify;color:#CE5937;">
    Multiplying the size of a sample by some factor divides the SE for a percentage not by the whole factor -- but by its square root.
</p>


---

When the sample size goes up, the SE for the number goes up, while the SE for the percentage goes down. That is because the SE for the number goes up slowly relative to sample size:

- The SE for the sample number goes up like the square root of the sample size
- The SE for the sample percentage goes down like the square of the sample size



### Using the Normal Curve

This section will review the **expected value** and **SE** for a *sample percentage*, and use the **normal curve** to compute chances.

$\color{Green}{\text{Example}}$ 1. In a certain town, the telephone company has 100,000 subscribers. It plans to take a simple random sample of 400 of them as part of a market research study. *According to Census data, 20% of the company’s subscribers earn over  \$50,000 a year.* The percentage of persons in the sample with incomes over \$50,000 a year will be around $\underline{} \underline{}$, give or take  $\underline{} \underline{}$ or so.

**Solution**. The first step is to make a box model. Taking a sample of 400 subscribers is like drawing 400 tickets at random from a box of 100,000 tickets. The drawing is done at random without replacement.

The problem involves classifying the people in the sample according to whether their incomes are more than \$50,000 a year or not, and then counting the ones whose incomes are above that level. So each ticket in the box should be marked by $1$ or $0$. **It is given that $20 \%$ of the subscribers earn more than \$50,000 a year**, so 20,000 of the tickets in box are marked $1$. The other 80,000 are marked $0$. The sample is like 400 draws from the box. And the number of people in the sample who earn more than \$50,000 a year is like the sum of the draws. That completes the box model.

Now we have to work on the sum of the draws from the 0-1 box. The expected value for the sum is $400 \times 0.2 = 80$. To compute the SE, we need the SD of the box. This is $\sqrt{0.2 \times 0.8} = 0.4$. There are 400 draws, so the SE for the sum is $\sqrt{400} \times 0.4 = 8$.

However, the question is about percent. We convert to percent relative to the size of the sample: 80 out of 400 is $20 \%$, and 8 out of 400 is $2 \%$. The expected value for the sample percentage is $20 \%$, and the SE is $2 \%$. That completes the solution: 

The percentage of persons in the sample with incomes over \$50,000 a year will be around $\underline{20 \%}$, give or take  $\underline{2 \%}$ or so.

$\color{Green}{\text{Example}}$ 2. (Continues example 1.) Estimate the chance that between $18 \%$ and $22 \%$ of the persons in the sample earn more than \$50,000 a year.

**Solution**. The expected value for the sample percentage is $20 \%$, and the SE is $2 \%$. Now convert to standard units:

![normal curve e5](./images/stats_normal_curve_e5.png)

That completes the solution.

**Here, the normal curve was used to figure chances. Why is that legitimate?**

There is a probability histogram for the number of high earners in the sample (figure 3). Areas in this histogram represent chances. This probability histogram follows the normal curve (top panel of figure 3). Conversion to percent is only a change of scale, so the probability histogram for the sample percentage (bottom panel) looks just like the top histogram -- and follows the curve too. In example 2, the curve was used on the probability histogram for the sample percentage, not on a histogram for data.

![normal curve e6](./images/stats_normal_curve_e6.png)

When do we change to a 0-1 box? To answer this question, think about the arithmetic being done on the sample values. The arithmetic might involve:

- adding up the sample values, to get an average, or
- classifying and counting, to get a percentage.

if the problem is about classifying and counting, put $0$'s and $1$'s in the box.



### The Correction Factor

Pollsters are trying to predict the results. There are about 1.5 million eligible voters in New Mexico, and about 15 million in the state of Texas. Suppose one polling organization takes a simple random sample of 2,500 voters in New Mexico, in order to estimate the percentage of voters in that state who are Democratic. Another polling organization takes a simple random sample of 2,500 voters from Texas. Both polls use exactly the same techniques. Both estimates are likely to be a bit off, by chance error. For which poll is the chance error likely to be smaller?

The New Mexico poll is sampling one voter out of 600, while the Texas poll is sampling one voter out of 6,000. It does seem that the New Mexico poll should be more accurate than the Texas poll. *However, this is one of the places where intuition comes into head-on conflict with statistical theory, and it is intuition which has to give way*. In fact, the accuracy expected from the New Mexico poll is just about the same as the accuracy to be expected from the Texas poll.

---

<p style="text-align: justify;color:#CE5937;">
    When estimating percentages, it is the absolute size of the sample which determines accuracy, not the size relative to the population. This is true if the sample is only a small part of the population, which is the usual case.
</p>


---

In fact, the draws are made without replacement. However, the number of draws is just a tiny fraction of the number of tickets in the box. Taking the draws without replacement barely changes the composition of the box.

In essence, that is why the size of the population has almost nothing to do with the accuracy of estimates. Still, there is a shade of difference between drawing with and without replacement. When drawing without replacement, the box does get a bit smaller, reducing the variability slightly. So the SE for drawing without replacement is a little less than the SE for drawing with replacement. There is a mathematical formula that says how much smaller:
$$
SE_{without\_replacement} = \text{correction factor} \times SE_{with\_replacement}
$$
The $\text{correction factor}$ itself somewhat complicated:
$$
\text{correction factor} = \sqrt{\frac{\text{number of tickets in box } - \text{number of draws}}{\text{number of tickets in box} - \text{one}}}.
$$
When the number of tickets in the box is large relative to the number of draws, the $\text{correction factor}$ is nearly $1$ and can be ignored. 

![correction factors](./images/stats_correction_factors.png)

Then it is the absolute size of the sample which determines accuracy, through the SE for drawing with replacement. The size of the population does not really matter. On the other hand, if the sample is a substantial fraction of the population, the correction factor must be used.

## The Accuracy of Percentages

The previous section **reasoned from the box to the draws**. Draws were made at random from a *box whose composition was known*, and a typical problem was finding the chance that the percentage of `1`’s among the draws would be in a
given interval. 

It is often very useful to **turn this reasoning around, going instead from the draws to the box**. A statistician would call this **inference** from the sample to the population. Inference is the topic of this section.

For example, suppose a survey organization wants to know the percentage of Democrats in a certain district. They might estimate it by taking a simple random sample. Naturally, the percentage of Democrats in the sample would be used to estimate the percentage of Democrats in the district—an example of reasoning backward from the draws to the box.

Because the sample was chosen at random, it is possible to say how accurate the estimate is likely to be, just from the size and composition of the sample. This section will explain how. The technique is one of the key ideas in statistical theory. It will be presented in the polling context.

$\color{Green}{\text{Example}}$ 3. A political candidate wants to enter a primary in a district
with 100,000 eligible voters, but only if he has a good chance of winning. He
hires a survey organization, which takes a simple random sample of 2,500 voters.
In the sample, 1,328 favor the candidate, so the percentage is ${1328 \over 2500} \times 100\% \approx 53\%$. The politician chuckled and said out loud that he'd win. The pollster reminded him that it is **just** the sample percentage, not the percentage in the whole district. The politician has arrived at the crucial question to ask when considering survey data: how far wrong is the estimate likely to be?

As the pollster wanted to say, the likely size of the chance error is given by the standard error. To figure that, a box model is needed.

![box model 2](./images/stats_box_model_2.png)

To get the SE for the sum, the survey organization needs the SD of the box.
This is
$$
SD = \sqrt{\text{(fraction of 1's)} \times \text{(fraction of 0's)}}.
$$
At this point, the pollsters seem to be stuck. They don’t know how each ticket in the box should be marked. They don’t even know the fraction of `1`’s in the box. That parameter represents the fraction of voters in the district who favor their candidate, which is exactly what they were hired to find out. (Hence the question marks in the box.)

Survey organizations lift themselves over this sort of obstacle by their own
*bootstraps*. They substitute the fractions observed in the sample for the unknown
fractions in the box. On this basis, the SD of the box is estimated as $\sqrt{0.53 \times 0.47} \approx 0.50$. The SE for the number of voters in the sample who favor the candidate is estimated as $\sqrt{2500} \times 0.50 = 25$. The 25 measures the likely size of the chance error in the 1,328. Now 25 people out of 2,500 (the size of the sample) is $1\%$.

This calculation shows that his pollster’s estimate of $53\%$ is only likely to be off by 1 percentage point or so. It is very unlikely to be off by as much as 3 percentage points -- that’s 3 SEs.

---

<p style="text-align: justify;color:#CE5937;">
    The bootstrap. When sampling from a 0-1 box whose composition is unknown, the SD of the box can be estimated by subsutituting the fractions of 0's and 1's in the sample for the unknown fractions in the box. The estimate is good when the sample is reasonably large.
</p>


---

The bootstrap procedure may seem crude. But even with moderate-sized samples, the fraction of `1`’s among the draws is likely to be quite close to the fraction in the box. Similarly for the `0`’s.

**One point is worth more discussion**. The expected value for the number of
`1`’s among the draws (translation—the expected number of sample voters who
favor the candidate) is
$$
2500 \times \text{fraction of 1's in the box}.
$$
This is unknown, because the fraction of `1`’s in the box is unknown. The SE of $25$ says about how far the 1328 is from its expected value. The 1328 is an *observed value*[^1]; the contrast is with the unknown expected value.



$\color{Green}{\text{Example}}$ 4. In fall 2005, a city university had 25,000 registered students. To estimate the percentage who were living at home, a simple random sample of 400 students was drawn. It turned out that 317 of them were living at home. Estimate the percentage of students at the university who were living at home in fall 2005. Attach a standard error to the estimate.

**Solution**. The sample percentage is
$$
{317 \over 400} \times 100\% \approx 79\%
$$
That is the estimate for the population percentage.

For the SE, a box model is needed.

![box model 3](./images/stats_box_model_3.png)

The fraction of `1`'s in the box is unknown, but can be estimated as $0.79$ -- the fraction observed in the sample. So the SD of the box is estimated by the bootstrap method as $\sqrt{0.79 \times 0.21} \approx 0.41$. The SE for the number of students in the sample who were living at home is estimated as $\sqrt{400} \times 0.41 \approx 8$. The $8$ gives the likely size of the chance error in the 317. Now convert to percent, relative to the size of the sample:
$$
{8 \over 400} \times 100\% \approx 2\%
$$
The SE for the sample percentage is estimated as $2\%$.

In the sample, $79\%$ of the students were living at home. The $79\%$ is off the mark by 2 percentage points or so. That is what the SE tells us.

### Confidence Interval

In example 4, the sample percentage was $79\%$. How far can the population percentage be from $79\%$?

The SE was estimated as $2\%$, suggesting a chance error of around $2\%$ in size. So the population percentage could be $77\%$  which means a chance error of $2\%$:
$$
\text{sample percentage} = \text{population percentage} + \text{chance error}
$$
The population percentage could also be $76\%$, corresponding to a chance error of $3\%$. This is getting unlikely, because $3\%$ represents 1.5 SEs. The population percentage could even be as small as $75\%$, but this is still more unlikely; $4\%$ represents 2 SEs. Of course, the population percentage could be on the other side of the sample percentage, corresponding to *negative chance errors*.

With chance errors, there is no sharp dividing line between the possible and the impossible. Errors larger in size than 2 SEs do occur—infrequently. What happens with a cutoff at 2 SEs? Take the interval from 2 SEs below the sample percentage to 2 SEs above:

![confidence interval e1](./images/stats_confidence_interval_e1.png)

The is a **confidence interval** for the population percentage, with a *confidence level* of about $95\%$. We can be about $95\%$ confident that the population percentage is caught inside the interval from $75\%$ to $83\%$.

What if you want a different confidence level? Anything except $100\%$ is possible, by going the right number of SEs in either direction from the sample percentage.

- $\text{sample percentage} \pm 1 \text{SE}$ is a $68\%$-confidence interval for the population percentage
- $\text{sample percentage} \pm 2 \text{SE}$ is a $95\%$-confidence interval for the population percentage
- $\text{sample percentage} \pm 3 \text{SE}$ is a $99.7\%$-confidence interval for the population percentage

There are no definite limits to the normal curve: no matter how large a finite interval you choose, the normal curve has some area outside that interval.

**Confidence levels** are often quoted as being “about” so much. There are two reasons. (i) The standard errors have been estimated from the data. (ii) The normal approximation has been used.

**If the normal approximation does not apply, neither do the methods of this section**. There is no hard-and-fast rule for deciding. The best way to proceed is to imagine that the population has the same percentage composition as the sample. Then try to decide whether the normal approximation would work for the sum of the draws from the box. For instance, a sample percentage near $0\%$ or $100\%$ suggests that the box is lopsided, so a large number of draws will be needed before the normal approximation takes over. On the other hand, if the sample percentage is near $50\%$, the normal approximation should be satisfactory when there are only a hundred draws or so.

### Interpreting a Confidence Interval

It seems more natural to say “There is a 95% chance that the population percentage is between 75% and 83%.”

But there is a problem here. 

<p style="text-align: justify;color:#CE5937;">
    In the frequency theory, a chance represents the percentage of the time that something will happen.
</p>


No matter how many times you take stock of all the students registered at that university in the fall of 2005, the percentage who were living at home back then will not change. Either this percentage was between $75\%$ and $83\%$, or not. There really is no way to define the chance that the parameter will be in the interval from $75\%$ to $83\%$. That is why statisticians have to turn the problem around slightly. They realize that the chances are in the sampling procedure, not in the parameter. And they use the new word “confidence” to remind you of this.

<p style="text-align: center;color:#CE5937;">
    The chances are in the sampling procedure, not in the parameter.
</p>


The confidence level of $95\%$ says something about the sampling procedure, and we are going to see what that is. 

- The confidence interval depends on the sample. If the sample had come out differently, the confidence interval would have been different.
- Confidence levels are a bit difficult, because they involve thinking not only about the actual sample but about other samples that could have been drawn.

<p style="text-align: justify;color:#CE5937;">
    A confidence interval is used when estimating an unknown parameter from sample data. The interval gives a range for the parameter, and a condifence level that the range covers the true value.
</p>


**Of course, investigators usually cannot tell whether their particular interval covers the population percentage, because they do not know that parameter**. But they are using a procedure that works $95\%$ of the time: take a simple random sample, and go 2 SEs either way from the sample percentage. It is as if their interval was drawn at random from a box of intervals, where $95\%$ cover the parameter and only $5\%$ are lemons.



A hundred survey organizations are hired to estimate the percentage of red marbles in a large box. **Unknown to the pollsters, this percentage is 80%**. Each organization takes a simple random sample of 2,500 marbles, and computes a 95%-confidence interval for the percentage of reds in the box.

The percentage of reds is different from sample to sample, and so is the estimated
standard error. As a result, the intervals have different centers and lengths. Some
of the intervals cover the percentage of red marbles in the box, others fail.

![confidence interval](./images/stats_confidence_interval.png)

**Probabilities are used when you reason forward, from the box to the draws; confidence levels are used when reasoning backward, from the draws to the box.** There is a lot to think about here, but keep the main idea of the chapter in mind:

<p style="text-align: justify;color:#CE5937;">
    A sample precentage will be off the population percentage, due to chance error. The SE tells you the likely size of the amount off.
</p>


Confidence levels were introduced to make this idea more quantitative.

### WARNING

<p style="text-align: justify;color:#CE5937;">
    The formulas for simple random samples may not apply to other kinds of samples.
</p>


Here is the reason. Logically, the procedures in this chapter all come out of the square root law[^2]. When the size of the sample is small relative to the size of the population, taking a simple random sample is just about the same as drawing at random with replacement from a box—the basic situation to which the square root law applies. 

The phrase “at random” is used here in its technical sense: *at each stage, every ticket in the box has to have an equal chance to be chosen*. If the sample is not taken at random, the square root law does not apply, and may give silly answers.



[^1]: Observed value are rarely more than 2 or 3 SEs away from the expected value.
[^2]: The square root law. When drawing at random with replacement from a box of numbered tickets, the standard error for the sum of the draws is $\sqrt{\text{number of draws} \times \text{(SD of box)}}$. NOTE that the square root law only applies to draws from a box.



## The Accuracy of Averages

The object of this chapter is to estimate the accuracy of an average computed from a simple random sample. This section deals with a preliminary question:

- How much chance variability is there in the average of numbers drawn from a box?