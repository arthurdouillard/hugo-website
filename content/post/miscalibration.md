+++
title = "How To Be Confident In Your Neural Network Confidence"
blog = true

date = 2019-05-30
draft = false

authors = []
hidden = false
tags = []
summary = "Why deep neural networks confidence are flawed and how to fix it."

thumbnail = "/figures/miscalibration.png"

[header]
image = ""
caption = ""
+++

Those notes are based on the research paper
"**On Calibration of Modern Neural Networks**" by [(Guo et al, 2017.)](https://arxiv.org/abs/1706.04599).

# How To Be Confident In Your Neural Network Confidence?

Very large and deep models, as ResNet, are far more accurate than their older counterparts, as LeNet, on computer vision datasets such as CIFAR100. **However while
they are better at classifying images, we are less confident in their own confidence!**

Most neural networks for classification uses as last activation a softmax: it
produces a distribution of probabilities for each target (cat, dog, boat, etc.).
These probabilities sum to one. We may expect that if for a given image, our
model associate a score of 0.8 to the target ‘boat’, our model is confident at
80% that this is the right target.

Over 100 images that were detected as boat, we can expect approximately that 80
images are indeed real boats, while the 20 remaining were false positives.

It was true for shallow model as LeNet but as newer models gained in accuracy
**their confidences became decorrelated from the “real confidence”**.

This does not work anymore for deep neural networks:

{{< figure src="/figures/miscalibration.png" caption="*Figure 1: Miscalibration in modern neural network [[source](https://arxiv.org/abs/1706.04599)]*" >}}

As you can see, older networks as LeNet had a low accuracy (55%) but their
confidence was actually in line with the accuracy! Modern networks as ResNet have
a higher accuracy (69%) but as showed in figure 1, they are over-confident.

This discrepancy between the model confidence and the actual accuracy is called
**miscalibration**.


## Why It Is Important

Outside of toy datasets used in the academy, it can be useful to know how much
confident our model is.

Imagine we have a model predicting frauds. We want to flag some transaction as
suspicious based on the model confidence that it is a fraud.
We could definitely compute an optimal threshold on the validation set, and then
every confidence above this threshold would be flagged as a fraud. However
this computed threshold could be 0.2 or 0.9 but would probably make much sense to a human.

A model without miscalibration would help the users to interpret better the
predictions.


## Why It Happens

The authors explores empirically what are the causes of this miscalibration in
modern networks.

They measure the miscalibration with the **E**xpected **C**alibration **E**rror (ECE):
the average difference between the confidence and the accuracy. This metric should
be minimized.

### Higher Capacity & Cross-Entropy


The most interpretable cause of the miscalibration is the increase of capacity
and the cross-entropy loss.

Model capacity can be seen as a measurement of how much a model can memorize.
With an infinite capacity, the model could simply learn by heart the whole
training dataset. A trade-off has to be made between a low and high capacity.
If it is too low the model wouldn’t be able to learn essential features of your
data. If it is too high, the model will learn too much and overfit instead of
generalize. Indeed comprehension is compression: by leaving few enough capacity
the model has to pick up the most representative features (pretty much in the
same way PCA works) and will then generalize better (but too few capacity & no
learning will happen!).

The new architectures such as ResNet have way more capacity than the older
LeNet (25M parameters for the former and 20k for the latter). This high
capacity led to better accuracy: the training set can almost be learned by heart.

In addition the models optimizes the cross-entropy loss that force them to be
right AND to be very confident. The higher capacity helped to lower the
cross-entropy loss and thus encourages deep neural networks to be over-confident.
As you’ve seen on figure 1, the new models are now over-confident.


{{< figure src="/figures/miscalibration_capacity.png" caption="*Figure 2: More capacity (in depth or width) increases the miscalibration. [[source](https://arxiv.org/abs/1706.04599)]*" >}}

### The Mysterious Batch Normalization

Batch Normalization normalizes the tensors in a network. It greatly improves the
training convergence & the final performance. Why exactly it works that well
is still a bit undefined ([see more](/posts/normalization)).

The authors remark empirically that using Batch Normalization increased the miscalibration
but could not find an exact reason why.

{{< figure src="/figures/miscalibration_bn.png" caption="*Figure 3: Batch Normalization increases the miscalibration. [[source](https://arxiv.org/abs/1706.04599)]*" >}}

Could the help given by this method in training facilitate the over-confidence?

### Regularization

The weight decay is an additional loss that penalizes the L2 norm of the weights.
The larger the weights, the bigger the norm and thus the loss. By constraining the weights
magnitude, it avoid the model finding extreme weight values that could make it overfit.

The authors found that increasing the regularization decreases the model accuracy
as expected. However it also decreased the miscalibration! The answer is then again
because regularization avoid overfitting & thus over-confidence.

{{< figure src="/figures/miscalibration_reg.png" caption="*Figure 4: More regularization decreases the miscalibration. [[source](https://arxiv.org/abs/1706.04599)]*" >}}

## How To Fix Miscalibration

This article's title, "*How To Be Confident In Your Neural Network Confidence*",
led you to believe that you would discover how to reduce miscalibration.

You're not going to reduce the capacity, remove Batch Normalization, and increase
the regularization: you'll hurt too much your precious accuracy.

Fortunately there are post-processing solutions. The authors describe several
but the most effective one is also the simplest: **Temperature Scaling**.

Instead of computing the softmax like this:

$$\text{softmax}(x)_i = \frac{e^{y_i}}{\Sigma_j^N e^{y_j}}$$

All the logits (values just before the final activation, here softmax) are divided
by the same value called temperature:

$$\text{softmax}(x)_i = \frac{e^{\frac{y_i}{T}}}{\Sigma_j^N e^{\frac{y_j}{T}}}$$

Similar to ([Hinton et al, 2015.](https://arxiv.org/abs/1503.02531)), this temperature
*softens the probabilities*.

Extreme probabilities (high confidence) are more decreased than smaller probabilities
(low confidence). The authors find the optimal temperature by minimizing the
Expected Calibration Error on the validation set.

The miscalibration is almost entirely corrected:

{{< figure src="/figures/miscalibration_tempscaling.png" caption="*Figure 5: Temperature Scaling fixes the miscalibration. [[source](https://arxiv.org/abs/1706.04599)]*" >}}



Another cool feature of Temperature Scaling: because all logits are divided by the
same value, and that softmax is a [monotone function](https://en.wikipedia.org/wiki/Monotonic_function),
the accuracy remains unchanged!

