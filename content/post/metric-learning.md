+++
title = "Lowshot learning using a metric"
blog = true

date = 2019-01-13
draft = true

authors = []

tags = []
summary = "When having few samples, lowshot situation, learning a metric can help."

thumbnail = ""

[header]
image = ""
caption = ""
+++

# 1. A problem

Deep Learning usually requires many samples: ImageNet has a thousand samples per
classes. Thanks to **transfer learning** (see [Huh et al, 2016](https://arxiv.org/abs/1608.08614)
for a good review) we can use the knowledge of large datasets (i.e. ImageNet),
and transfer it to a smaller dataset.

This has two drawbacks:

1. The new small dataset must similar to the larger dataset, you can expect
the knowledge learned on dogs & cats to generalize well on clothing for example.
This is a problem of **domain adaptation** that I'll hopefuly cover in another post.

2. Furthermore, even if transfer learning can alleviate the need of
data, we still need too much! A good rule of thumb is having at least 500-1,000
images per class.

The second problem is called **lowshot learning**. It is a commonly faced problem
at my work, [Heuritech](https://www.heuritech.com/), when we want to classify
bag models having only a dozen of images to train on.

*Note that lowshot is also called fewshots or meta-learning*.

# 2. Toy datasets & lowshot vocabulary

MNIST is the famous *toy* dataset for deep learning featuring handwritten
numbers. It is made of 10 classes (0 to 9), with 7,000 images per class.

Its equivalent in the [Omniglot dataset](https://github.com/brendenlake/omniglot).
It features 1623 unique characters taken from several alphabets (Latin, Korean, etc.).
The real difference is that each unique character has only 20 samples!

{{< figure src="/figures/omniglot_0.png" caption="*Figure 1: A sample of the Omniglot dataset.*" >}}

A particular vocabulary, introduced by [Koch et al, 2015](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) was used to described the
task of lowshot:

During training, the goal is to differentiate all unique characters of the
<span style="color: red;">**background set**</span>. On the next figure, in red,
the model must learn that the Latin *F* is different from the Korean *B/P*.

Then during the test phase, we will classify the unseen classes of the
<span style="color: blue;">**query set**</span> using the few labelled images of the
<span style="color: green;">**support set**</span>.

{{< figure src="/figures/omniglot_1.png" caption="*Figure 2: The vocabulary of lowshot tasks.*" >}}


# 2. Several methods for lowshot learning

Several kinds of methods exist to solve lowshot problems. The three main ones were
nicely presented by Oriol Vinyals at NeurIPS 2017:

{{< figure src="/figures/meta_ml.png" caption="*Figure 1: Taxonomy of lowshot learning, [[source]](http://metalearning-symposium.ml/files/vinyals.pdf)*" >}}

I'll present in this post only those metric-based.


# 3. Learning a metric

The key idea of metric-based models is to learn a distance or a similarity between
samples. Then having the pairwise metric computed between all labeled & unlabeled
samples, and finally do a nearest-neighbours to classify the unlabeled data
based on their similarity with the labelled data:

{{< figure src="/figures/metric-learning.jpg" caption="*Figure 2: Unlabeled data is classified based on its distance with the labeled data.*" >}}

On the previous figure, a single instance of the red class is enough to to classify
its three closest unlabelled instances as also red. It's a case of *one-shot learning*.


# 4. Various models

## 4.1. Siamese, Triplet, & co.

[Bromley et al, 1993](https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf)
introduced **Siamese Network** for signature verification. The network sees a pair
of signatures & determines whether they are the same or one is a forgery. A
classification would be far too data-demanding and couldn't scale to millions unique signatures.

[(Koch et al, 2015)](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) re-used
this architecture to Omniglot with the task described previously. A same network
extracts features from each image of the pair and pass those to a classifier
followed by a sigmoid. Then the pair of features are substracted together & an
absolute function is applied on the result. This result is fed again to a classifier
with a single output neuron & a sigmoid, predicting a similarity score: 1 if the
pair corresponds to the same character else 0.

As the authors remarks, they didn't use a L1 distance on the features that would
collapse into a single value. Instead the final classifier is weighting the
importance of the component-wise distance.

An implementation of this architecture can be done in a few lines of Pytorch:

```python
class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = MyConvNet()
        self.fc1 = nn.Linear(out_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, 1)

    def forward(self, image1, image2):
        x1 = self.convnet(image1)
        x2 = self.convnet(image2) # Shape (batch, channel, width, height)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1) # Shape (batch, channel * width * height)

        x1 = torch.sigmoid(self.fc1(x1))
        x2 = torch.sigmoid(self.fc1(x2)) # Shape (batch, channel * width * height)

        x = torch.abs(x1 - x2) # Shape (batch, channel * width * height)
        x = torch.sigmoid(self.fc2(x)) # Shape (batch, 1)

        return x
```

The loss is simply a **binary cross-entropy**. What the Siamese Network does is
pulling closer in the features space images of the same label, while pushing apart
those of different labels.

**PLOT GRAPH**

An older alternative to [(Koch et al, 2015)](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
is a Siamese Network, with a L2 distance instead of a component-wise L1 distance,
and the **contrastive loss** [(hadsell-chopra-lecun, 2006)](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf):

$L = (1 - Y) D^2 + Y (\max(0, m - D))^2$

$Y$ is equal to 1 if the pair is dissimilar, otherwise 0. The first part of the
loss ($(1 - Y) D^2)

- $


Siamese

Triplet [(Hoffer & Ailon, 2014)](https://arxiv.org/abs/1412.6622)

Triplet ranking [(Ye & Guo, 2018)](https://arxiv.org/abs/1804.07275)

Quadruplet [(Chen et al, 2017)](https://arxiv.org/abs/1704.01719)

N-pair [(Sohn et al, 2016)](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)


## 4.3. Imprinted weights

[(Qi et al, 2017)](https://arxiv.org/abs/1712.07136)

## 4.4. Histogram Loss

[(Ustinova & Lempitsky, 2016)](https://arxiv.org/abs/1611.00822)

## 4.4. Matching Network

[(Vinyals et al, 2016)](https://arxiv.org/abs/1606.04080)

