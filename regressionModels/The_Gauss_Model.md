## The Gauss Model

The box model for measurement error will now be described in more detail.

The basic situation is that a series of *repeated measurements* are made on some quantity. According to the model, each measurement differs from the exact value by a chance error; this error is like a draw made at random from a box of tickets -- the *error box*. Successive measurements are done independently and under the same conditions, so the draws from the error box are made with replacement. **To capture the idea that the chance errors aren't systematically positive or systematically negative, it is assumed that the average of the numbers in the error box equals 0**. This model is named after Carl Friedrich Gauss (Germany, 1777-1855), who worked on measurement error in astronomical data.

<p style="text-align:justify;color:lightblue;">
    In the Gauss model, each time a measurement is made, a ticket is drawn at random with replacement from the error box. The number on the ticket is the chance error. It is added to the exact value to give the actual measurement. The average of the error box is equal to 0.
</p>



In the model, it is the SD of the box which gives the likely size of the chance errors. Usually, this SD is unknown and must be estimated from the data.

Take the 100 measurements on NB 10[^1], for example. According to the model, each measurement is around the exact weight, but it is off by a draw from the error box:
$$
\begin{eqnarray}
\text{1st measurement} &=& \text{exact weight + 1st draw from error box} \\
\text{2nd measurement} &=& \text{exact weight + 2nd draw from error box} \\
&\vdots& \\
\text{100th measurement} &=& \text{exact weight + 100th draw from error box}
\end{eqnarray}
$$
With the NB 10 data, the SD of the draws would be a fine estimate for the SD of the error box. The catch is that the draws cannot be recovered from the data, because the exact weight is unknown. However, the variability in the measurements equals the variability in the draws, because the exact weight does not change from measurement to measurement. Moreover, adding the exact value to all the errors does not change the SD[^2]. That is why statisticians use the SD of the measurements when computing the SE.

<p style="text-align:justify;color:lightblue">
    When Gauss model applies, the SD of series of repeated measurements can be used to estimate the SD of the error box. The estimate is good when there are enough measurements.
</p>


There may be another way to get at the SD of the error box. When there is a lot of experience with the measurement process, it is better to estimate the SD from all the past data rather than a few current measurements. The reason: the error box belongs to the measurement process, not the thing being measured.

The version of the Gauss model presented here makes the assumption that there is no bias in the measuring procedure. When bias is present, each measurement is the sum of three term:
$$
\text{exact value + bias + chance error.}
$$
Then the SE for the average no longer says how far the average of the measurements i from the exact value, but only how far it is from
$$
\text{exact value + bias.}
$$
The methods of this chapter are no help in judging bias. In some cases, bias can be negligible. In other situations, bias can be more serious than chance errors, and harder to detect.



## CONCLUSION

NB 10 is just a chunk of metal. It is weighted on a contraption of platforms, gears, and levers. The results of these weights have been subjected to a statistical analysis involving the **standard error, the normal curve, and confidence intervals.**

It is the the Gauss model which connects the mathematics to NB 10.

The chance errors are like draws from a box; their average is like the average of the draws. The number of draws is so large that the probability histogram for the average will follow the normal curve very closely. Without the model there would be no box, no standard error, and no confidence levels.

**Statistical inference** uses chance methods to draw conclusions from data. Attaching a standard error to an average is an example. Now it is always possible to go through the SE procedure mechanically. Many computer programs will do the work for you. It is even possible to label the output as a "standard error". Do not get hypnotized by the arithmetic or the terminology. The procedure only makes sense because of the **square root law**[^3]. 

<p style="text-align:justify;color:lightblue">
    The implicit assumption is that the data are like the results of drawing from a box.<br>
    Statistical inference can be justified by putting up an explicit chance model for the data. No box, no inference.
</p>



The *descriptive statistics* -- drawing diagrams or calculating numbers which summarize data and bring out the salient features. Such techniques can be used very generally, because they do not involve any hidden assumptions about where the data came from. For statistical inference, however, models are basic.



[^1]: Called NB 10  because it is owned by the National Bureau and its nominal value is 10 grams. (The exact weight will be a little different due to the chance error, hence the "nominal".)
[^2]: SD says how far away numbers on a list are from their average. $ \text{SD} =  \text{r.m.s (deviation from average)}$.
[^3]: The square root law. When drawing at random with replacement from a box of numbered tickets, the standard error for the sum of the draws is $\sqrt{\text{number of draws} \times \text{(SD of box)}}$. NOTE that the square root law only applies to draws from a box.