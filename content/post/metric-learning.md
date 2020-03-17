+++
title = "Fewshots with metric learning"
blog = true

date = 2019-01-13
draft = true

authors = []

tags = []
summary = "Deep Learning requires thousand of samples. By learning a metric this constraint is eased up to classify with a dozen samples."

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

1. The new small dataset must similar to the larger dataset: you cannot expect
the knowledge learned on dogs & cats to generalize well on clothing.
This is a problem of **domain adaptation** that I'll hopefuly cover in another post.

2. Even if transfer learning can alleviate the need of data, we still need too much of it: a good rule of thumb is having at least 500-1,000 images per class in this case.

Putting aside the first drawbacks, we want to learn with only a dozen of samples.
Transfer learning is then not enough.

The field of learning with few samples is called **fewshot learning**.

# 2. Toy datasets & fewshot vocabulary

MNIST is the famous *toy* dataset for deep learning featuring handwritten
numbers. It is made of 10 classes (0 to 9), with 7,000 images per class.

Its equivalent in the [Omniglot dataset](https://github.com/brendenlake/omniglot).
It features 1623 unique characters taken from several alphabets (Latin, Korean, etc.).
The challenge is that each unique character has only 20 samples!

{{< figure src="/figures/omniglot_0.png" caption="*Figure 1: A sample of the Omniglot dataset.*" >}}

A particular vocabulary, introduced by [Koch et al, 2015](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) was used to describe the task of fewshot:

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

Metric learning is widely used in **retrieval search** were the goal is to find the most
similar items given a query item: aka finding similar shoes on e-commerce website based
on my photography of a shoe.

# 4. Various models

This list of models is far from being exaustive. I focus mainly on deep learning
models that were used for lowshot but be aware that this is only a subset of metric
learning solutions.

## 4.1. Siamese, Triplet, & co.

### 4.1.1 Siamese networks

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
class Siamese1(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = MyConvNet()
        self.fc1 = nn.Linear(out_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, 1)

    def forward(self, images1, images2):
        x1 = self.convnet(images1)
        x2 = self.convnet(images2)  # Shape (batch, channel, width, height)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)  # Shape (batch, channel * width * height)

        x1 = torch.sigmoid(self.fc1(x1))
        x2 = torch.sigmoid(self.fc1(x2))  # Shape (batch, channel * width * height)

        x = torch.abs(x1 - x2)  # Shape (batch, channel * width * height)
        x = self.fc2(x)  # Shape (batch, 1)

        return x

def loss(logits, similarities):
    return F.binary_cross_entropy_with_logits(logits, similarities)
```

The loss is simply a **binary cross-entropy**. What the Siamese Network does is
pulling closer in the features space images of the same label, while pushing apart
those of different labels.

**PLOT GRAPH**

An older alternative to [(Koch et al, 2015)](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
is a Siamese Network, with a L2 distance instead of a component-wise L1 distance,
and the **contrastive loss** [(hadsell-chopra-lecun, 2006)](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf):

```python
class Siamese2(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = MyConvNet()
        self.fc1 = nn.Linear(out_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, 1)

    def forward(self, images1, images2):
        x1 = self.convnet(images1)
        x2 = self.convnet(images2)  # Shape (batch, channel, width, height)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)  # Shape (batch, channel * width * height)

        distance = torch.norm(x1 - x2, 2)  # Shape (batch,)

        return distance

def constrastive_loss(distances, similarities, margin=0.2):
    return (1 - similarities) * (distances ** 2) + similarities * (torch.clamp(margin - distances, min=0.) ** 2)
```


$L = (1 - Y) D^2 + Y (\max(0, m - D))^2$

$Y$ is equal to 1 if the pair is dissimilar, otherwise 0. The first part of the
loss ($(1 - Y) D^2)

### 4.1.2. Triplet networks

Triplet [(Hoffer & Ailon, 2014)](https://arxiv.org/abs/1412.6622)

Triplet ranking [(Ye & Guo, 2018)](https://arxiv.org/abs/1804.07275)

### 4.1.3. Quadruplet networks

Quadruplet [(Chen et al, 2017)](https://arxiv.org/abs/1704.01719)

Learning a Distance Metric from Relative Comparisons between Quadruplets of Images

### 4.1.4. N-pair networks

After incrementing from pair, triplet, and quadruplet, n-pair generalizes to N images each
compared to each other.

N-pair [(Sohn et al, 2016)](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)

angular loss

## 4.3. Imprinted weights

[(Qi et al, 2017)](https://arxiv.org/abs/1712.07136)

## 4.4. Histogram Loss

[(Ustinova & Lempitsky, 2016)](https://arxiv.org/abs/1611.00822)

## 4.4. Matching Network

In the same vein of n-pair network, matching network were conceived to compare i

[(Vinyals et al, 2016)](https://arxiv.org/abs/1606.04080)


Prototype network

Matching prototypes networks


cosine classifier (A Closer Look at Few-shot Classification)
