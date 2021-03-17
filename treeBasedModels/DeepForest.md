# [Deep Forest](https://arxiv.org/abs/1702.08835) 



> In this paper, we extend our preliminary study which proposes the [gcForest]() (multi-Grained Cascade Forest) approach for constructing deep forest, a non-NN style deep model. This is a novel decision tree ensemble, with a cascade structure which enables representation learning by forests. Its representational learning ability can be further enhanced by multi-grained scanning, potentially enabling gcForest to be contextual or structural aware. The cascade levels can be automatically determined such that the model complexity can be determined in a data-dependent way rather than manually designed before training; this enables gcForest to work well even on small-scale data, and enables users to control training costs according to computational resource available. Moreover, the gcForest has much fewer hyper-parameters than DNNs. Even better news is that its performance is quite robust to hyper-parameter settings; our experiments show that in most cases, it is able to get excellent performance by using the default setting, even across different data from different domains.



## Inspiration from DNNs[^1]

It is widely recognized that the *representation learning* ability is crucial for the success of deep neural networks. We believe[^2] that the model complexity itself cannot explain the success of DNNs (e.g., large flat networks are not as successful as deep ones) and the *layer-by-layer processing* is what really crucial for representation learning in DNNs. Figure 1 provides an illustration[^17].

![layer-by-layer_processing](./images/df_crucial_of_representation_learning.png)

Learning models, e.g., decision trees and Boosting machines, which also conduct layer-by-layer processing, why they are not as successful as DNNs? We believe that the most important distinguishing factor is that, in contrast to DNNs where new features are generated as illustrated in Figure 1, decision trees and Boosting machines always work on the original feature representation without creating new features during the learning process, or in other words, there is no in-model feature transformation. Moreover, DTs and Boosting machines can only have limited model complexity.

Overall, we conjecture that behind the mystery of DNNs there are three crucial characteristics, i.e., layer-by-layer processing, in-model feature transformation, and sufficient model complexity. We will try to endow these characteristics to our non-NN style deep model.

## Inspiration from Ensemble Learning

It is well known that an ensemble (multiple learners are trained and combined) can usually achieve better generalization performance than single learners.

To construct a good ensemble, the individual learners should be *accurate* and *diverse*. Combining only accurate learners is often inferior to combining some accurate learners with some relatively weaker ones, because the complementarity is more important than pure accuracy. Here is the equation derived from *error-ambiguity decomposition*[^32]:
$$
\tag{1}
E = \bar{E} - \bar{A},
$$
where $E$ denotes the error of an ensemble, $\bar{E}$ denotes the average error of individual classifiers in the ensemble, and $\bar{A}$ denotes the average *ambiguity*, later called *diversity*, among the individual classifiers. Eq. 1 reveals that, the more accurate and more diverse the individual classifiers, the better the ensemble. However, it could not be taken as an objective function for optimization, because the *ambiguity* term is mathematically defined in the derivation and cannot be operated directly[^32]. Actually, *"what is diversity?"* remains the holy grail problem in ensemble learning.

In practice, the basic strategy of diversity enhancement is to inject randomness based on some heuristics during the training process. Roughly speaking, there are four major category of mechanisms[^63]:

1. **data sample manipulation**: which works by generating different data samples to trian individual learners. 

   E.g., bootstrap sampling is exploited by Bagging; sequential importance sampling is adopted by AdaBoost.

2. **input feature manipulation**: which works by generating different feature subspaces to train individual learners.

   E.g., the Random Subspace approach randomly picks a subset of features.

3. **learning parameter manipulation**: which works by using different parameter settings of the base learning algorithm to generate diverse individual learners.

   E.g., different initial selections can be applied to individual neural networks; different split selections can be applied to individual decision trees.

4. **output representation manipulation**: which works by using different output representations to generate diverse individual learners.

   E.g., the ECOC approach employs error-correcting output codes; the Flipping Output method randomly changes the labels of some training instances.

Note that, however, these mechanisms are not always effective. More information about ensemble learning can be found in the book Ensemble Methods[^63].

Next we give you, the gcForest, which can be viewed as a decision tree ensemble approach that utilizes almost all categories of mechanisms for diversity enhancement.

## The gcForest Approach

First introduce the cascade forest structure, and then the multi-grained scanning, followed by the overall architecture.

### Cascade Forest Structure

Representation learning in DNNs mostly relies on the layer-by-layer processing of raw features. Inspired by this recognition, gcForest employs a cascade structure, as illustrated in Figure 2, where each level of cascade receives feature information processed by its preceding level, and outputs its processing result to the next level.

![cascade-forest structure](./images/df_cascade_forest_structure.png)

Each level is an ensemble of decision tree forests, i.e., an *ensemble* of *ensembles*. Here, we include different types of forests to encourage the *diversity*, because diversity is crucial for ensemble construction.

For simplicity, suppose that we use two completely-random tree forests and two random forests. Each completely-random tree forest contains 500 completely-random trees, generated by randomly selecting a feature for split at each node of the tree, and growing tree until pure leaf, i.e., each leaf node contains only the same class of instances. Similarly, each random forest contains 500 trees, by randomly selecting $\sqrt{d}$ number of features as candidate ($d$ is the number of input features) and choosing the one with the best *gini* value for split. (Note that the number of trees in each forest is a hyper-parameter.)

Given an instance, each forest will produce an estimate of class distribution, by counting the percentage of different classes of training examples at the leaf node where concerned instance falls, and then averaging across all trees in the same forest, as illustrated in Figure 3, where red color highlights paths along which the instance traverses to leaf nodes.

![class-vector generation](./images/df_class-vector_generation.png)

The estimated class distribution forms a class vector, which is then concatenated with the original feature vector to be input to the next level of cascade. For example, suppose there are three classes, then each of the four forests will produce a three-dimensional class vector; thus the next level of cascade will receive 12 ($= 3 \times 4$) augmented features.

Note that here we take the simplest form of class vectors, i.e., the class distribution at the leaf nodes into which the concerned instance falls. The more complex form of class vectors can be constructed by getting more distributions such as class distribution of the parent nodes which express prior distribution, the sibling nodes which express complementary distribution, etc.

To reduce the risk of over-fitting, class vector produced by each forest is generated by $k$-fold cross validation. In detail, each instance will be used as training data for $k - 1$ times, resulting $k - 1$ class vectors, which are then averaged to produce the final class vector as augmented features for the next level of cascade. After expanding a new level, the performance of the whole cascade can be estimated on validation set, and the training procedure will terminate if there is no significant performance gain; thus, the number of cascade levels is automatically determined. Note that the training error rather than cross validation error can also be used to control the cascade growth when the training cost is concerned or limited computation resource available.

### Multi-Grained Scanning

DNNs are powerful in handling feature relationships, e.g., convolutional-NN are effective on image data where spatial relationships among the raw pixels are critical; recurrent-NN are effective on sequence data where sequential relationships are critical. Inspired by this recognition, we enhance cascade forest with a procedure of multi-grained scanning.

![sliding-windows scanning](./images/df_sliding-windows_scanning.png)

As Figure 4 illustrates, sliding windows are used to scan the raw features. Suppose there are 400 raw features and a window size of 100 features is used. For sequence data, a 100-dimensional feature vector will be generated by sliding the window for one feature; in total 301 feature vectors are produced. If the raw features are with spacial relationships, such as a $20 \times 20$ panel of 400 image pixels, then a $10 \times 10$ window will produce 121 feature vectors. All feature vectors extracted from positive/negative training examples are regarded as positive/negative instances, which will then be used to generate class vectors like in Section [Cascade Forest Structure](###Cascade Forest Structure): the instance extracted from the same size of windows will be used to train a completely-random tree forest and a random forest, and then the class vectors are generated and concatenated as transformed features. As Figure 4 illustrates, suppose that there are 3 classes and a 100-dimensional window is used; then, 301 three-dimensional class vectors are produced by each forest, leading to a 1860-dimensional transformed feature vector corresponding to the original 400-dimensional raw feature vector.

***

**convolution operations**: padding and strides

when the input shape is  $(n_h \times n_w)$, the *convolution kernel*'s shape is $(k_h \times k_w)$, 

**with no padding and stride (default with $s_h = s_w = 1$)**, then output shape will be:
$$
(n_h - k_h + 1, n_w - k_w + 1),
$$
**with padding (add $p_h$ rows and $p_w$ columns ) and stride $(s_h = s_w = 1)$,** then output shape will be:
$$
(n_h - k_h + p_h + 1, n_w - k_w + p_w + 1),
$$
**with padding (add $p_h$ rows and $p_w$ columns ) and stride $(s_h, s_w)$,** then output shape will be:
$$
\bigg((n_h-k_h+p_h+s_h)/s_h, (n_w-k_w+p_w+s_w)/s_w \bigg)
$$


If we set $p_h=k_h-1$ and $p_w=k_w-1$, then the output shape will be simplified to:
$$
\bigg((n_h+s_h-1)/s_h, (n_w+s_w-1)/s_w \bigg)
$$


Going a step further, if the input height and width are divisible by the strides on the height and width, then the output shape will be:
$$
\bigg((n_h/s_h)， (n_w/s_w)\bigg)
$$

***

For the instances extracted from the windows, we simply assign them with the label of the original training example. Here, some label assignments are inherently incorrect. For example, suppose the original training example is a positive image about "car"; it is clearly that many extracted instances do not contain a car, and therefore, they are incorrectly labeled as positive. This is actually related to the Flipping Output method[^4], a representative of output representation manipulation for ensemble diversity enhancement.

Note that when transformed feature vectors are too long to be accommodated, feature sampling can be performed, e.g., by subsampling the instances generated by sliding window scanning, since completely-random trees do not rely on feature split selection whereas random forests are quite insensitive to inaccurate feature split selection. Such a feature sampling process is also related to the Random Subspace method[^24], a representative of input feature manipulation for ensemble diversity enhancement.

Figure 4 shows only one size of sliding window. By using multiple sizes of sliding windows, differently grained feature vectors will be generated, as show in Figure 5.

![multi-grained scanning](./images/df_multi-grained_scanning.png)






































[^1]: DNNs, in a more technically view, is "multiple layers of parameterized differentiable nonlinear modules that can be trained by back-propagation." Also note that back-propagation requires differentiability.
[^2]: There is no rigorous justification yet.



[^4]: L. Breiman. Randomizing outputs to increase prediction accuracy. Machine Learning, 40(3):113–120, 2000.



[^17]: I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT Press, Cambridge, MA, 2016.



[^24]: T. K. Ho. The random subspace method for constructing decision forests. IEEE Trans. Pattern Analysis and Machine Intelligence, 20(8):832–844, 1998.



[^32]: A. Krogh and J. Vedelsby. Neural network ensembles, cross validation, and active learning. In G. Tesauro, D. S.Touretzky, and T. K. Leen, editors, Advances in Neural Information Processing Systems 7, pages 231{238. 1995.
[^63]: Z.-H. Zhou. Ensemble Methods: Foundations and Algorithms. CRC, Boca Raton, FL, 2012.

