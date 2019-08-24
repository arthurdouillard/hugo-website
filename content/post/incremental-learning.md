+++
title = "Introduction to Incremental Learning"
blog = true

date = 2019-04-26
draft = true

authors = []

tags = []
summary = "Learning tasks incrementally using knowledge distillation & external memory."

thumbnail = ""

[header]
image = ""
caption = ""
+++

## Transfer Learning

**Transfer learning** allows to transfer the knowledge gained on one task (e.g
*ImageNet*) to another task (e.g *classify cats & dogs*). Usually the backbone
(a ConvNet in Computer Vision, like ResNet) is kept while a new classifier is
plugged in on top of it. During transfer, we train the new classifier &
**fine-tune** the backbone.

However by fine-tuning the model's backbone on the new task, it will probably change
it so much that it will forget the old task!

Neural Networks are good to learn, but bad to retain knowledge from previous task.

This is called a **catastrophic forgetting** [(French, 1995)](https://www.sciencedirect.com/science/article/pii/S1364661399012942). To solve it, one must be an optimal trade-off between
**rigidity** (being good on old classes) and **plasticity** (being good on new classes).

## Lifelong-Learning

This lack of knowledge retention is a problem in two cases:

- (Ideal) Robots must learn continuously without forgetting
-. New tasks (or new samples) keep coming but we don't want:
    - to store the increasingly bigger dataset
    - to retrain on all the data in order to not forget anything

The goal of **lifelong-learning** is thus three folds:

- Learn new tasks or new samples without forgetting the previous ones
- The model is kept between tasks in order to make us of the incremental knowledge
- All the seen data cannot be stored, either keep nothing or a constant-size subset.

Lifelong-learning can also be divided in three scenarios
[(Lomonaco and Maltoni, 2017)](https://arxiv.org/abs/1705.03550):

- **New instances** added with potentially new domains (*online learning*)
- **New classes** added (*incremental learning*)
- **New instances & new classes** added

In this article we will focus only on the incremental aspect. Note however that the
methods used are fairly similar between scenarios.

## Small literature review

[(Parisi et al, 2018)](https://arxiv.org/abs/1802.07569) defines 3 broad strategies:

- **External Memory** storing previous tasks data
- **Regularization** methods avoiding forgetting on previous tasks
- **Model Plasticity** extending the capacity

#### External Memory

Lifelong-learning has for constraint to have a size-bounded training dataset. The
role of the external memory is thus to sample only a subset of every old tasks.

Three variants exists:

- **Reharsal learning** keeps a subsample of previous data
- **Pseudo-Reharsal Learning** generates data using previous data's distribution
- **No memory**, training only on new data

#### Regularization

Regularization's goal in lifelong-learning is to force the model trained on task $T + 1$
to be similar enough to its previous version trained on task $T$. So that the
catastrophic forgetting is alleviated.

There are several ways to do so:

- The most obvious is to add to the loss a L2 distance between the weights of the
old & new version of the model. [(Kirkpatrick et al, 2017)](https://arxiv.org/abs/1612.00796)
and [(Zenke, Poole, and Ganguli, 2017)](https://arxiv.org/abs/1703.04200) add an
**importance factor** to make this distance more or less "elastic". An important
weight would be allowed to change less than a lesser important weight.
- [(Li and Hoiem, 2018)](https://arxiv.org/abs/1606.09282), [(Rebuffi et al, 2017)](https://arxiv.org/abs/1611.07725), and [(Castro et al, 2018)](https://arxiv.org/abs/1807.09536) add
a constraint on the probabilities. If the previous model attributed a confidence of $56%$
on the class $C$ for the sample $S$ then the new model should attribute it a close confidence.
- Finally [(Lopez-Paz and Ranzato, 2017)](https://arxiv.org/abs/1706.08840) imposes
a constraint on the loss itself by forbidding to increase on seen samples.

#### Model plasticity

Most lifelong-learning algorithms add new neurons to their classifier in order
to predict the new classes. [(Golkar, Kagan and Cho, 2019)](https://arxiv.org/abs/1903.04476)
doesn't and instead share the same classifier for every task, each task uses
a mask choosing the input neurons.

[(Yoon et al, 2018)](https://arxiv.org/abs/1708.01547) rationalizes that the total
model capacity can not suffice sometimes with the increasing number of tasks. Its
Dynamically Expandable Networks can copy existing neurons or even add new neurons
to solve this.

[(Frankle and Carbin, 2018)](https://arxiv.org/abs/1803.03635)'s Lottery Ticket
Hypothesis states that neural networks are vastly overparametrized & that we
can prune them down to a small sub-network having similar performance.
[(Fernando et al, 2017)](https://arxiv.org/abs/1701.08734)'s PathNet explores those
sub-networks using an evolutionary algorithms & uses one sub-network per task.
[(Golkar, Kagan and Cho, 2019)](https://arxiv.org/abs/1903.04476) imitates PathNet
but instead find the sub-networks using an in-training neurons sparsification
with a L1 regularization on the neurons activity ([see intuition](https://stats.stackexchange.com/a/159379)).

## Our focus

I'll detail now three papers, each an incremental (see the pun?) improvement over the
precedent:

- Learning without Forgetting [(Li and Hoiem, 2016)](https://arxiv.org/abs/1606.09282).
- iCaRL: Incremental Classifier and Representation Learning [(Rebuffi et al, 2017)](https://arxiv.org/abs/1611.07725)
- End-to-End Incremental Learning [(Castro et al, 2018)](https://arxiv.org/abs/1807.09536)

You can find my PyTorch implementations [here](https://github.com/arthurdouillard/incremental_learning.pytorch).

#### Evaluation

The three papers are evaluated on the iCIFAR100 benchmark: a training on CIFAR100
with a growing number of classes. New classes are batched in group of 2, 5, 10, 20,
and 50 producing respectively 50, 20, 10, 5, and 2 tasks (CIFAR100 has eponymously
100 classes).

At each task, the model is evaluated on all the seen classes.

Finally the plotted curve is not simply the accuracy, but what [(Rebuffi et al, 2017)](https://arxiv.org/abs/1611.07725) named the **average incremental accuracy**:

> The result of the evaluation are curves of the classification accuracies after
> each batch of classes. If a single number is preferable, we report the average
> of these accuracies, called average incremental accuracy.

That I understand as: "the accuracy at task $T + X$ is actually task $T + X$ accuracy
averaged with all previous tasks accuracy". Which would make sense, else two models
reaching the same accuracy on task $T + X$ would be equal while one could have
had consistently a better accuracy on previous tasks.

#### Learning without Forgetting

[(Li and Hoiem, 2016)](https://arxiv.org/abs/1606.09282)'s LwF does not use any
external memory. Meaning that once the model was trained on a task, the task's
data is completely discarded.

During the first task, the training is common: softmax + cross-entropy to
classify the images. However before every other tasks, we save the current model
predictions' confidence of old classes on the new task data.
These predictions are done on completely unseen data, that do not contain any old
classes. Still the confidences provide plenty of information: as
[(Hinton et al, 2015)](https://arxiv.org/abs/1503.02531) nicknamed a **dark knowledge**.

Starting from the second task, there are now two losses:

- The **classification loss** which is a cross-entropy between the new classes'
predictions & the ground-truth
- The **distillation loss** which is a binary-cross-entropy between the current
predictions of the old classes & its previously recorded version.

The goal of the **distillation loss** is to force the model to match the same
confidence on old classes as it did before the current task.

*Note that as [(Hinton et al, 2015)](https://arxiv.org/abs/1503.02531) suggested,
they add a temperature (the logits are raised to the power of $\frac{1}{\text{temperature}}$)
to soften the probabilities.*

#### iCaRL: Incremental Classifier and Representation Learning

iCaRL also use classification & distillation losses. The only difference