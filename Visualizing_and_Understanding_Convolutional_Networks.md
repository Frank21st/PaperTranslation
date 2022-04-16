**Visualizing and Understanding Convolutional Neural Networks** 

**Matthew  D.  Zeiler                                                 [zeiler@cs.nyu.edu](mailto:zeiler@cs.nyu.edu)**  Dept. of Computer Science, Courant Institute, New York University

**Rob  Fergus                                                      [fer](mailto:fergus@cs.nyu.edu)**[gus@cs.nyu.edu](mailto:gus@cs.nyu.edu)  Dept. of Computer Science, Courant Institute, New York University





## Abstract  摘要

Large Convolutional Neural Network models have recently demonstrated impressive classification performance on the ImageNet bench- mark ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)). However there is no clear understanding of why they perform so well, or how they might be im- proved. In this paper we address both issues. We introduce a novel visualization technique that gives insight into the function of intermediate feature layers and the operation of the classifier. We also perform an ablation study to discover the performance con- tribution from different model layers. This enables us to find model architectures that outperform Krizhevsky *et al.* on the Ima- geNet classification benchmark. We show our ImageNet model generalizes well to other datasets: when the softmax classifier is retrained, it convincingly beats the current state-of-the-art results on Caltech-101 and Caltech-256 datasets.

近来大型卷积神经网络模型在ImageNet benchmark ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)) 中证明了令人印象深刻的分类性能。但是，为什么该模型表现如此好，或者，怎么样才能更进一步改进提升该模型， 仍然没有明确的理解和认知。本文研究了这两个问题。本文介绍了一种新的可视化技术，该技术可以观察模型中间的特征层功能/作用和分类器的操作。本文也进行了消融实验（ablation study）以研究模型中不同层对性能的贡献。这使得本文发现了在ImageNet benchmark中比（Krizhevsky *et al）表现更好的模型架构。本文模型在其他数据集上泛化能力较好：当softmax classifier被重新训练后，模型打败了当前在Caltech-101 和 Caltech-256 数据集上表现最好的模型。

 

## 1. Introduction 介绍

Since their introduction by ([LeCun et al.](#bookmark36), [1989](#bookmark36)) in the early 1990’s, Convolutional Neural Networks (con- vnets) have demonstrated excellent performance at tasks such as hand-written digit classification and face detection. In the last year, several papers have shown that they can also deliver outstanding performance on more challenging visual classification tasks. ([Ciresan](#bookmark22) [et al.](#bookmark22), [2012](#bookmark22)) demonstrate state-of-the-art performance on NORB and CIFAR-10 datasets. Most notably, ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)) show record beating perfor- mance on the ImageNet 2012 classification benchmark, with their convnet model achieving an error rate of 16.4 %, compared to the 2nd place result of 26.1%. Several factors are responsible for this renewed inter- est in convnet models: (i) the availability of much larger training sets, with millions of labeled exam- ples; (ii) powerful GPU implementations, making the training of very large models practical and (iii) bet- ter model regularization strategies, such as Dropout ([Hinton et al.](#bookmark31), [2012](#bookmark31)).

Despite this encouraging progress, there is still lit- tle insight into the internal operation and behavior of these complex models, or how they achieve such good performances. From a scientific standpoint, this is deeply unsatisfactory. Without clear understanding of how and why they work, the development of better models is reduced to trial-and-error. In this paper we introduce a visualization technique that reveals the in- put stimuli that excite individual feature maps at any layer in the model. It also allows us to observe the evolution of features during training and to diagnose potential problems with the model. The visualization technique we propose uses a multi-layered Deconvo- lutional Network (deconvnet), as proposed by ([Zeiler](#bookmark26) [et al.](#bookmark26), [2011](#bookmark26)), to project the feature activations back to the input pixel space. We also perform a sensitivity analysis of the classifier output by occluding portions of the input image, revealing which parts of the scene are important for classification.

Using these tools, we start with the architecture of ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)) and explore different archi- tectures, discovering ones that outperform their results on ImageNet. We then explore the generalization abil- ity of the model to other datasets, just retraining the softmax classifier on top. As such, this is a form of su- pervised pre-training, which contrasts with the unsu- pervised pre-training methods popularized by ([Hinton](#bookmark30) [et al.](#bookmark30), [2006](#bookmark30)) and others ([Bengio et al.](#bookmark17), [2007](#bookmark17); [Vincent](#bookmark23) [et al.](#bookmark23), [2008](#bookmark23)).



 



### 1.1.  Related Work

Visualizing features to gain intuition about the net- work is common practice, but mostly limited to the 1st layer where projections to pixel space are possible. In higher layers this is not the case, and there are limited methods for interpreting activity. ([Erhan et al.](#bookmark27), [2009](#bookmark27)) find the optimal stimulus for each unit by perform- ing gradient descent in image space to maximize the unit’s activation. This requires a careful initialization and does not give any information about the unit’s in- variances. Motivated by the latter’s short-coming, ([Le](#bookmark35) [et al.](#bookmark35), [2010](#bookmark35)) (extending an idea by Berkes and Wiskott ([Berkes & Wiskott](#bookmark19), [2006](#bookmark19))) show how the Hessian of a given unit may be computed numerically around the optimal response, giving some insight into invariances. The problem is that for higher layers, the invariances are extremely complex so are poorly captured by a simple quadratic approximation. Our approach, by contrast, provides a non-parametric view of invariance, showing which patterns from the training set activate the feature map.

## 2. Approach

We use standard fully supervised convnet models throughout the paper, as defined by (LeCun et al.,1989) and (Krizhevsky et al., 2012). These models map a color 2D input image xi, via a series of layers, to a probability vector yˆ over the C different i classes. Each layer consists of (i) convolution of the previous layer output (or, in the case of the 1st layer, the input image) with a set of learned filters; (ii) pass- ing the responses through a rectified linear function (relu(x) = max(x,0)); (iii) [optionally] max pooling over local neighborhoods and (iv) [optionally] a lo- cal contrast operation that normalizes the responses across feature maps. For more details of these opera- tions, see (Krizhevsky et al., 2012) and (Jarrett et al., 2009). The top few layers of the network are conven- tional fully-connected networks and the final layer is a softmax classifier. Fig. 3 shows the model used in many of our experiments.





### 2.1. Visualization with a Deconvnet

Understanding the operation of a convnet requires in- terpreting the feature activity in intermediate layers. We present a novel way to *map these activities back to the input pixel space*, showing what input pattern orig- inally caused a given activation in the feature maps. We perform this mapping with a Deconvolutional Net- work (deconvnet) ([Zeiler et al.](#bookmark26), [2011](#bookmark26)). A deconvnet can be thought of as a convnet model that uses the same components (filtering, pooling) but in reverse, so instead of mapping pixels to features does the oppo- site. In ([Zeiler et al.](#bookmark26), [2011](#bookmark26)), deconvnets were proposed as a way of performing unsupervised learning. Here, they are not used in any learning capacity, just as a probe of the convnet.

To examine a convnet, a deconvnet is attached to each of its layers, as illustrated in Fig. [1](#bookmark1)(top), providing a continuous path back to image pixels. To start, an input image is presented to the convnet and features computed throughout the layers. To examine a given convnet activation, we set all other activations in the layer to zero and pass the feature maps as input to the attached deconvnet layer.  Then we  successively

(i) unpool, (ii) rectify and (iii) filter to reconstruct the activity in the layer beneath that gave rise to the chosen activation.  This is then repeated until input



ers, to a probability vector



*y*ˆ*i* over the *C* different



pixel space is reached.



classes. Each layer consists of (i) convolution of the

previous layer output (or, in the case of the 1st layer, the input image) with a set of learned filters; (ii) pass- ing the responses through a rectified linear function (*relu*(*x*) = max(*x,* 0)); (iii) [optionally] max pooling over local neighborhoods and (iv) [optionally] a lo- cal contrast operation that normalizes the responses across feature maps. For more details of these opera- tions, see ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)) and ([Jarrett et al.](#bookmark32), [2009](#bookmark32)). The top few layers of the network are conven- tional fully-connected networks and the final layer is a softmax classifier. Fig. [2](#bookmark3) shows the model used in many of our experiments.



​                 *{*  *}*                

We train these models using a large set of *N* labeled images *x, y* , where label *y**i* is a discrete variable indicating the true class. A cross-entropy loss func- tion, suitable for image classification, is used to com- pare *y*ˆ*i* and *y**i*. The parameters of the network (fil- ters in the convolutional layers, weight matrices in the fully-connected layers and biases) are trained by back- propagating the derivative of the loss with respect to the parameters throughout the network, and updating the parameters via stochastic gradient descent. Full details of training are given in Section [3.1](#bookmark2).





**Unpooling:** In the convnet, the max pooling opera- tion is non-invertible, however we can obtain an ap- proximate inverse by recording the locations of the maxima within each pooling region in a set of *switch* variables. In the deconvnet, the unpooling operation uses these switches to place the reconstructions from the layer above into appropriate locations, preserving the structure of the stimulus. See Fig. [1](#bookmark1)(bottom) for a visualization of the procedure.

**Rectiftcation:** The convnet uses *relu* non-linearities, which rectify the feature maps. To obtain valid recon- structions, we pass the reconstructed signal through a *relu* non-linearity.

**Filtering:** The convnet uses learned filters to con- volve the feature maps from the previous layer. To invert this, the deconvnet uses transposed versions of the same filters, but applied to the rectified maps, not the output of the layer beneath. In practice this means flipping each filter vertically and horizontally.

Projecting down from higher layers uses the switch settings generated by the max pooling in the convnet on the way up. As these switch settings are peculiar to a given input image, the reconstruction obtained from a single activation thus resembles a small piece

 



of the original input image, with structures weighted according to their contribution toward to the feature activation. Since the model is trained discriminatively, they implicitly show which parts of the input image are discriminative. Note that these projections are *not* samples from the model, since there is no generative process involved.

 



nected layers, given that they contain the majority of the model’s parameters. However, decreasing both severely affects performance, showing the importance of having a minimum depth to the model. Altering the number of units in the fully connected layers (2048 or 8192 vs 4096) makes little difference to performance. Increasing the size of the convolutional layers 3,4,5 to 512-1024-512 maps, from 384-384-256, does give a gain in performance, but the model starts to over-fit due to the big increase in number of parameters. The over-



 

 

 



fitting is more pronounced when increasing the size of





​                               

both the convolutional and fully connected layers.





 

 

Unpooling





 

 

Max Locations “Switches”

 

Unpooled





 

 

 

 

 

Rectified





Pooled Maps

 

Pooling





 

*Table 1.* ImageNet 2012 classification error rates with var- ious architectural changes to our ImageNet model.

The experiments in Table [1](#bookmark0) show that by increasing



Maps



Feature Maps



the number of feature maps in the middle layers, the

model of ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)) may be improved



*Figure 1.* Top: A deconvnet layer (left) attached to a con- vnet layer (right). The deconvnet will reconstruct an ap- proximate version of the convnet features from the layer beneath. Bottom: An illustration of the unpooling oper- ation in the deconvnet, using *switches* which record the location of the local max in each pooling region (colored zones) during pooling in the convnet.

## 3. Experiments

We start by training a large convolutional network model on the ImageNet dataset, using the exact ar- chitecture specified in ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)) and attempt to replicate their result on the validation set. The ImageNet dataset ([Deng et al.](#bookmark24), [2009](#bookmark24)) consists  of 1.3M/65k/100k training/validation/test examples, spread over 1000 categories. Details of the training procedure are given in Section [3.1](#bookmark2) below. As shown in Table [2](#bookmark4), we achieve error rate within 0*.*1% of their reported value on the ImageNet 2012 validation set.

We now explore a range of different model architec- tures in an attempt to understand the relative impor- tance of each layer. In Table [1](#bookmark0), we modify the size of (a) the convolutional layers, (b) the fully connected layers and (c) both sections of the model. Decreas- ing each part separately only results in a modest per- formance drop. This is surprising for the fully con-



upon. Fig. [2](#bookmark3) shows the best performing architecture, which has a dramatically larger layers 3,4 and 5. When evaluated on the Imagenet 2012 validation set, it sig- nificantly outperforms ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)), beat- ing their single model result by 1*.*8% (see Table [2](#bookmark4)). When we combine multiple models, we obtain a test error of **15***.***3**%, which matches the absolute best per- formance on this dataset, despite only using the much smaller 2012 training set. We note that this error is almost half that of the top non-convnet entry in the ImageNet 2012 classification challenge, which obtained 26*.*1% error.

### 3.1. Training Details

The models were trained on the ImageNet 2012 train- ing set (1.3 million images, spread over 1000 different classes). Each RGB image was preprocessed by resiz- ing the smallest dimension to 256, cropping the center 256x256 region, subtracting the per-pixel mean (across all images) and then using 10 different sub-crops of size 224x224 (corners + center with(out) horizontal flips). Stochastic gradient descent with a mini-batch size of 128 was used to update the parameters, starting with a learning rate of 10*−*2, in conjunction with a momen- tum term of 0*.*9. Dropout ([Hinton et al.](#bookmark31), [2012](#bookmark31)) is used in the fully connected layers (6 and 7) with a rate of







Stride 4





3x3 max pool stride 2



96

contrast norm.



3x3 max pool stride 2





contrast norm.



3x3 max pool stride 2





4096

units





4096

units



C

class softmax



 

Input Image



3   27



5

1     96



13     3

1





256



6       256



Layer 1            Layer 2



Layer 3     Layer 4        Layer 5



Layer 6 Layer 7



Output



 

*Figure 2.* Architecture of our 8 layer convnet model. A 224 by 224 crop of an image (with 3 color planes) is presented as the input. This is convolved with 96 different 1st layer filters (red), each of size 11 by 11, using a stride of 4 in both x and y. The resulting feature maps are then: (i) passed through a rectified linear function (not shown), (ii) pooled (max within 3x3 regions, using stride 2) and (iii) contrast normalized across feature maps to give 96 different 27 by 27 element feature maps. Similar operations are repeated in layers 2,3,4,5. The last two layers are fully connected, taking features



​                                         Error %                  Val      Top-1                  Val      Top-5                  Test      Top-5                            ([Krizhevsky et al.](#bookmark34),      [2012](#bookmark34)), 1 convnet      ([Krizhevsky et al.](#bookmark34),      [2012](#bookmark34)), 5 convnets      ([Krizhevsky et al.](#bookmark34),      [2012](#bookmark34)), 1  convnets*      ([Krizhevsky et al.](#bookmark34),      [2012](#bookmark34)), 7  convnets*                  40*.*7      38*.*1      39*.*0      36*.*7                  18*.*2      16*.*4      16*.*6      15*.*4                  *−−*      16*.*4      *−−*      15*.*3                            Our replication of      ([Krizhevsky et al.](#bookmark34),      [2012](#bookmark34)), 1 convnet                         40*.*5                         18*.*1                  *−−*                            1 convnet as per      Fig. [2](#bookmark3)                  38*.*3                  16*.*4                  16*.*5                            5 convnets as per      Fig. [2](#bookmark3)                  36*.*6                  15*.*3                  15*.*3                                    

from the top convolutional layer as input in vector form (6 *·* 6 *·* 256 = 9216 dimensions). The final layer is a *C*-way softmax function, *C* being the number of classes. All filters and feature maps are square in shape.



 



 

 

 

 

*Table* *2.* ImageNet 2012 classification error rates.  The  *∗*

indicates models that were trained on both ImageNet 2011

and 2012 training sets with an additional convolution layer.

0.5. We manually anneal the learning rate throughout training, decreasing it when the validation set error flatlines. All weights are initialized to 10*−*2 and biases are set to 0.

As in ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)), we produce multiple different crops and flips of each training example to boost training set size. We stopped training after 70 epochs, which took around 24 days on a single GTX580 GPU, using an implementation based on ([Krizhevsky](#bookmark34) [et al.](#bookmark34), [2012](#bookmark34)). One further difference is that the sparse connections used in Krizhevsky’s layers 3,4,5 (due to the model being split across 2 GPUs) are replaced with dense connections in our models.

## 4. Convnet Visualization

Using the model described in Section [3.1](#bookmark2), we now use the deconvnet to visualize the feature activations on the ImageNet validation set.

### 4.1. Feature Evolution during Training

Fig. [3](#bookmark5) visualizes the progression during training of the strongest activation (across all training examples) within a given feature map projected back to pixel space. Sudden jumps in appearance result from a change in the image from which the strongest acti- vation originates. Due to space constraints, only a randomly selected subset of feature maps are visual-



ized and zooming is needed to see the details clearly. As expected, the first layer filters consist of Gabors and low-frequency color. The 2nd layer features are more complex, corresponding to conjunctions of edges and color patterns. The 3rd layer features show larger image parts. Within a given feature projection, signif- icant variations in contrast can be seen, showing which parts of the image contribute most to the activation and thus are most discriminative, e.g. the lips and eyes on the persons face (Row 12). The visualization from the 4th and 5th layer show activations that respond to complex objects. Note that little of the scene back- ground is reconstructed, since it is irrelevant to pre- dicting the class.

### 4.2. Feature Invariance

Fig. [8](#bookmark15) shows feature visualizations from our model once training is complete. However, instead of show- ing the single strongest activation for a given feature map, we show the top 9 activations. Projecting each separately down to pixel space reveals the different structures that excite a given feature map, hence show- ing its invariance to input deformations. Alongside these visualizations we show the corresponding image patches. These have greater variation than visualiza- tions as the latter solely focus on the discriminant structure within each patch. For example, in layer 5, row 1, col 2, the patches appear to have little in com- mon, but the visualizations reveal that this particular feature map focuses on the grass in the background, not the foreground objects.

The projections from each layer show the hierarchi- cal nature of the features in the network. Layer 2 re- sponds to corners and other edge/color conjunctions. Layer 3 has more complex invariances, capturing sim- ilar textures (e.g. mesh patterns (Row 1, Col 1); text



 

![img]() ![img]() ![img]() ![img]()

*Figure 3.* Evolution of model features through training. Each layer’s features are displayed in a different block. Within each block, we show a randomly chosen subset of features at epochs [1,2,5,10,20,30,40,64]. The visualization shows the strongest activation (across all training examples) for a given feature map, projected down to pixel space using our deconvnet approach. Color contrast is artificially enhanced and the figure is best viewed in electronic form.

 



(R2,C4)). Layer 4 shows significant variation, but is more class-specific: dog faces (R1,C1); bird’s legs (R4,C2). Layer 5 shows entire objects with significant pose variation, e.g. keyboards (R1,C11) and dogs (R4).

Fig. [4 ](#bookmark7)shows 5 sample images being translated, rotated and scaled by varying degrees while looking at the changes in the feature vectors from the top and bot- tom layers of the model, relative to the untransformed feature. Small transformations have a dramatic effect in the first layer of the model, but a lesser impact at the top feature layer, being quasi-linear for translation & scaling. The network output is stable to translations and scalings, but not to rotation.

### 4.3. Occlusion Sensitivity

With image classification approaches, a natural ques- tion is if the model is truly classifying the object alone, or if it is using the surrounding context. Fig. [5](#bookmark10) at- tempts to answer this question by systematically oc- cluding different portions of the input image with a

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image045.png)





![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image046.jpg)

 

*Figure 6.* Images used for correspondence experiments. Col 1: Original image. Col 2,3,4: Occlusion of the right eye, left eye, and nose respectively. Other columns show examples of random occlusions.

 

 

 



​                 *i*                

​                    *i*                                        *i*                                        *i*                                        *i*                   same part of the face in each image (e.g. all left eyes, see Fig. [6](#bookmark6)). For each image *i*, we then com- pute:  *s**l*  =  *x**l* *−* *x*˜*l*,  where *x**l*  and  *x*˜*l*  are  the feature





sifier. The examples clearly show the model is local- izing the objects within the scene, as the probability of the correct class drops significantly when the ob-

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image047.png)



vectors at layer *l* for the original and occluded im-



​                 *i,j*=1*,i*                

​                    *j*                                        *i*                                        *j*                                        Σ                   ages respectively. We then measure the consistency of this difference vector *s* between all related image pairs (*i, j*):  ∆*l* =    5  *H*(sign(*s**l*)*,* sign(*s**l* )), where *H*



*
*

the strongest feature map of the top convolution layer, in addition to activity in this map as a function of occluder position. When the occluder covers the im- age region that appears in the visualization, we see a strong drop in activity in the feature map. This shows that the visualization genuinely corresponds to the im- age structure that stimulates that feature map, hence validating the other visualizations in Fig. [3 ](#bookmark5)and Fig. [8](#bookmark15).

### 4.4. Correspondence Analysis

Deep models differ from many existing recognition approaches in that there is no explicit mechanism for establishing correspondence between specific ob- ject parts in different images (e.g. eyes and noses for faces). However, an intriguing possibility is that deep models might be *implicitly* computing them. To explore this, we take 5 randomly drawn dog images with frontal pose and systematically mask out the



is Hamming distance. A lower value indicates greater

consistency in the change resulting from the masking operation, hence tighter correspondence between the same object parts in different images. In Table [3](#bookmark8) we compare the ∆ score for three parts of the face (left eye, right eye and nose) to random parts of the ob- ject, using features from layer *l* = 5 and *l* = 7. The lower score for these parts, relative to random object regions, for the layer 5 features show the model does establish some degree of correspondence.

 

## 5. Feature Generalization

The experiments above show the importance of the convolutional part of our ImageNet model in obtain- ing state-of-the-art performance. This is supported by the visualizations of Fig. [8 ](#bookmark15)which show the complex in- variances learned in the convolutional layers. We now



 

 

![img]()![img]()10

 

9

 

8

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image049.png)7

 

6

 

5

 

4

 

3

 

2

 

1

 

0

–60     –40     –20       0       20      40      60

Vertical Translation (Pixels)

![img]()![img]()12

 

 

10

 

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image051.png)8

 

 

6

 

 

4

 

 

2





 

0.8

 

![img]()0.7

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image051.png)0.6

 

0.5

 

0.4

 

0.3

 

0.2

 

0.1

 

0

–60     –40     –20       0       20      40      60

Vertical Translation (Pixels)

![img]()0.7

 

 

0.6

 

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image051.png)0.5

 

 

0.4

 

 

0.3

 

 

0.2

 

 

0.1





 

1

 

![img]()0.9

 

0.8

 

0.7

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image052.png)0.6

 

0.5

 

0.4

 

0.3

 

0.2

 

0.1

 

0

–60     –40     –20       0       20      40      60

Vertical Translation (Pixels)

![img]()1

 

0.9

 

0.8

 

0.7

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image052.png)0.6

 

0.5

 

0.4

 

0.3

 

0.2

 

0.1



 

0

1         1.2        1.4        1.6        1.8

Scale (Ratio)

![img]()![img]()15

 

 

 

 

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image051.png)10

 

 

 

 

 

5





0

1         1.2        1.4        1.6        1.8

Scale (Ratio)

![img]()1.4

 

 

1.2

 

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image051.png)1

 

 

0.8

 

 

0.6

 

 

0.4

 

 

0.2





0

1         1.2        1.4        1.6        1.8

Scale (Ratio)

![img]()1

 

0.9

 

0.8

 

0.7

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image054.png)0.6

 

0.5

 

0.4

 

0.3

 

0.2

 

0.1



 

0

0      50 100 150 200 250 300 350

Rotation Degrees





0

0      50 100 150 200 250 300 350

Rotation Degrees





0

0      50 100 150 200 250 300 350

Rotation Degrees



*Figure 4.* Analysis of vertical translation, scale, and rotation invariance within the model (rows a-c respectively). Col 1: 5 example images undergoing the transformations. Col 2 & 3: Euclidean distance between feature vectors from the original and transformed images in layers 1 and 7 respectively. Col 4: the probability of the true label for each image, as the image is transformed.

 



 

| Occlusion Location | Mean  Feature  Sign  Change Layer 5 | Mean  Feature  Sign  Change Layer 7 |
| ------------------ | ----------------------------------- | ----------------------------------- |
| Right Eye          | 0*.*067 *±* 0*.*007                 | 0*.*069 *±* 0*.*015                 |
| Left Eye           | 0*.*069 *±* 0*.*007                 | 0*.*068 *±* 0*.*013                 |
| Nose               | 0*.*079 *±* 0*.*017                 | 0*.*069 *±* 0*.*011                 |
| Random             | 0*.*107 *±* 0*.*017                 | 0*.*073 *±* 0*.*014                 |

 

*Table* *3.* Measure of correspondence for different object parts in 5 different dog images. The lower scores for the eyes and nose (compared to random object parts) show the model implicitly establishing some form of correspondence of parts at layer 5 in the model. At layer 7, the scores are more similar, perhaps due to upper layers trying to discriminate between the different breeds of dog.

 

explore the ability of these feature extraction layers to generalize to other datasets, namely Caltech-101 ([Fei-](#bookmark28) [fei et al.](#bookmark28), [2006](#bookmark28)), Caltech-256 ([Griffin et al.](#bookmark29), [2006](#bookmark29)) and PASCAL VOC 2012. To do this, we keep layers 1-7 of our ImageNet-trained model fixed and train a new softmax classifier on top (for the appropriate number of classes) using the training images of the new dataset. Since the softmax contains relatively few parameters, it can be trained quickly from a relatively small num- ber of examples, as is the case for certain datasets.

This approach is a *supervised* form of pre-training, since the bulk of the model parameters have been learned in a supervised fashion on the ImageNet data. This prevents direct comparisons to existing algo- rithms since they did not use the ImageNet data dur-



ing training. However, the results do give an absolute assessment of the performance of features extracted by our network. We also try a second strategy of training a model from scratch, i.e. resetting layers 1-7 to ran- dom values and train them, as well as the softmax, on the training images of the dataset.

 

**Caltech-101:** We follow the procedure of ([Fei-fei](#bookmark28) [et al.](#bookmark28), [2006](#bookmark28)) and randomly select 15 or 30 images per class for training and test on up to 50 images per class reporting the average of the per-class accuracies in Ta- ble [4](#bookmark9), using 5 train/test folds. Training took 17 min- utes for 30 images/class. The pre-trained model beats the best reported result for 30 images/class from ([Bo](#bookmark21) [et al.](#bookmark21), [2013](#bookmark21)) by 2.2%. The convnet model trained from scratch however does terribly, only achieving 46.5%.

 

| # Train                                                      | Acc %  15/class                   | Acc %  30/class                  |
| ------------------------------------------------------------ | --------------------------------- | -------------------------------- |
| ([Bo et al.](#bookmark21), [2013](#bookmark21))  ([Jianchao et   al.](#bookmark33), [2009](#bookmark33)) | *−*  73*.*2                       | 81*.*4 *±* 0*.*33  84*.*3        |
| Non-pretrained convnet                                       | 22*.*8 *±* 1*.*5                  | 46*.*5 *±* 1*.*7                 |
| ImageNet-pretrained  convnet                                 | **83***.***8**  *±* **0***.***5** | **86***.***5** *±* **0***.***5** |

*Table* *4.* Caltech-101 classification accuracy for our con- vnet models, against two leading alternate approaches.

**Caltech-256:** We follow the procedure of ([Griffin](#bookmark29) [et al.](#bookmark29), [2006](#bookmark29)), selecting 15, 30, 45, or 60 training images per class, reporting the average of the per-class accura- cies in Table [5](#bookmark11). Our ImageNet-pretrained model beats the state-of-the-art results obtained by ([Bo et al.](#bookmark21), [2013](#bookmark21)) by a significant margin: 74.2% vs 55.2% for 60 training images/class. However, as with Caltech-101,



 



 

(a) Input Image       (b) Layer 5, strongest feature map



(c) Layer 5, strongest feature map projections



(d) Classifier, probability of correct class



(e) Classifier, most probable class



![img]()

*Figure 5.* Four test examples where we systematically cover up different portions of the scene with a gray square (1st column) and see how the top (layer 5) feature maps ((b) & (c)) and classifier output ((d) & (e)) changes. (b): for each position of the gray scale, we record the total activation in one layer 5 feature map (the one with the strongest response in the unoccluded image). (c): a visualization of this feature map projected down into the input image (black square), along with visualizations of this map from other images. The first row example shows the strongest feature to be the dog’s face. When this is covered-up the activity in the feature map decreases (blue area in (b)). (d): a map of correct class probability, as a function of the position of the gray square. E.g. when the dog’s face is obscured, the probability for “pomeranian” drops significantly. (e): the most probable label as a function of occluder position. E.g. in the 1st row, for most locations it is “pomeranian”, but if the dog’s face is obscured but not the ball, then it predicts “tennis ball”. In the 2nd example, text on the car is the strongest feature in layer 5, but the classifier is most sensitive to the wheel. The 3rd example contains multiple objects. The strongest feature in layer 5 picks out the faces, but the classifier is sensitive to the dog (blue region in (d)), since it uses multiple feature maps. In the 4th example, the front wheel is the strongest feature, but the output depends on many parts of the vehicle.



the model trained from scratch does poorly. In Fig. [7](#bookmark12), we explore the “one-shot learning” ([Fei-fei et al.](#bookmark28), [2006](#bookmark28)) regime. With our pre-trained model, just 6 Caltech- 256 training images are needed to beat the leading method using 10 times as many images. This shows the power of the ImageNet feature extractor.

 

| # Train                                                      | Acc  %  15/class                 | Acc  %  30/class                 | Acc  %  45/class                  | Acc  %  60/class                  |
| ------------------------------------------------------------ | -------------------------------- | -------------------------------- | --------------------------------- | --------------------------------- |
| ([Sohn et al.](#bookmark18), [2011](#bookmark18))  ([Bo et al.](#bookmark21), [2013](#bookmark21)) | 35*.*1  40*.*5 *±* 0*.*4         | 42*.*1  48*.*0 *±* 0*.*2         | 45*.*7  51*.*9 *±* 0*.*2          | 47*.*9  55*.*2 *±* 0*.*3          |
| Non-pretr.                                                   | 9*.*0 *±* 1*.*4                  | 22*.*5 *±* 0*.*7                 | 31*.*2 *±* 0*.*5                  | 38*.*8 *±* 1*.*4                  |
| ImageNet-pretr.                                              | **65***.***7** *±* **0***.***2** | **70***.***6** *±* **0***.***2** | **72***.***7**  *±* **0***.***4** | **74***.***2**  *±* **0***.***3** |

*Table 5.* Caltech 256 classification accuracies.



75

 

![img]()70

 

65

 

60

 

![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image073.png)55

 

50

 

45

 

40

 

35

 

30

 

25

0      10     20     30     40     50     60

Training Images per−class



 

**PASCAL 2012:** We used the standard training and validation images to train a 20-way softmax on top of the ImageNet-pretrained convnet. This is not ideal, as PASCAL images can contain multiple objects and our model just provides a single exclusive prediction for



*Figure 7.* Caltech-256 classification performance as the number of training images per class is varied. Using only   6 training examples per class with our pre-trained feature extractor, we surpass best reported result by ([Bo et al.](#bookmark21), [2013](#bookmark21)).



 





​                                  Acc %                  [A]                  [B]                  Ours                  Acc %                  [A]                  [B]                  Ours                            Airplane                  92.0                  **97.3**                  96.0                  Dining tab                  63.2                  **77.8**                  67.7                            Bicycle                  74.2                  **84.2**                  77.1                  Dog                  68.9                  83.0                  **87.8**                            Bird                  73.0                  80.8                  **88.4**                  Horse                  78.2                  **87.5**                  86.0                            Boat                  77.5                  85.3                  **85.5**                  Motorbike                  81.0                  **90.1**                  85.1                            Bottle                  54.3                  **60.8**                  55.8                  Person                  91.6                  **95.0**                  90.9                            Bus                  85.2                  **89.9**                  85.8                  Potted pl                  55.9                  **57.8**                  52.2                            Car                  81.9                  **86.8**                  78.6                  Sheep                  69.4                  79.2                  **83.6**                            Cat                  76.4                  89.3                  **91.2**                  Sofa                  65.4                  **73.4**                  61.1                            Chair                  65.2                  **75.4**                  65.0                  Train                  86.7                  **94.5**                  91.8                            Cow                  63.2                  **77.8**                  74.4                  Tv                  77.4                  **80.7**                  76.1                            Mean                  74.3                  **82.2**                  79.0                  # won                  0                  **15**                  5                                    

each image. Table [6 ](#bookmark13)shows the results on the test set. The PASCAL and ImageNet images are quite differ- ent in nature, the former being full scenes unlike the latter. This may explain our mean performance being 3*.*2% lower than the leading ([Yan et al.](#bookmark25), [2012](#bookmark25)) result, however we do beat them on 5 classes, sometimes by large margins.



 

 

 

 

 

 

 

 

 

*Table 6.* PASCAL 2012 classification results, comparing our Imagenet-pretrained convnet against the leading two methods ([A]= ([Sande et al.](#bookmark16), [2012](#bookmark16)) and [B] = ([Yan et al.](#bookmark25), [2012](#bookmark25))).

### 5.1. Layer-by-Layer Performance Breakdown

We explore how discriminative the features in each layer of our Imagenet-pretrained model are. We do this by varying the number of layers retained from the ImageNet model and place either a linear SVM or softmax classifier on top. Table [7](#bookmark14) shows results on Caltech-101 and Caltech-256. For both datasets, a steady improvement can be seen as we ascend the model, with best results being obtained by using all layers. This supports the premise that as the feature hierarchies become deeper, they learn increasing pow- erful features.

 

|              | Cal-101  (30/class)              | Cal-256  (60/class)              |
| ------------ | -------------------------------- | -------------------------------- |
| SVM  (1)     | 44*.*8 *±* 0*.*7                 | 24*.*6 *±* 0*.*4                 |
| SVM  (2)     | 66*.*2 *±* 0*.*5                 | 39*.*6 *±* 0*.*3                 |
| SVM  (3)     | 72*.*3 *±* 0*.*4                 | 46*.*0 *±* 0*.*3                 |
| SVM  (4)     | 76*.*6 *±* 0*.*4                 | 51*.*3 *±* 0*.*1                 |
| SVM  (5)     | **86***.***2** *±* **0***.***8** | 65*.*6 *±* 0*.*3                 |
| SVM  (7)     | **85***.***5** *±* **0***.***4** | **71***.***7** *±* **0***.***2** |
| Softmax  (5) | 82*.*9 *±* 0*.*4                 | 65*.*7 *±* 0*.*5                 |
| Softmax (7)  | **85***.***4** *±* **0***.***4** | **72***.***6** *±* **0***.***1** |

*Table* *7.* Analysis of the discriminative information con- tained in each layer of feature maps within our ImageNet- pretrained convnet. We train either a linear SVM or soft- max on features from different layers (as indicated in brack- ets) from the convnet. Higher layers generally produce more discriminative features.

 

## 6. Discussion

We explored large convolutional neural network mod- els, trained for image classification, in a number ways. First, we systematically modified the network archi- tecture to reveal that having a minimum depth to the network, rather than any individual section, is vital to



the model’s performance. Expanding the size of these layers results in models who performance beats that of ([Krizhevsky et al.](#bookmark34), [2012](#bookmark34)). We then presented a novel way to visualize the activity within the model. This reveals the features to be far from random, uninter- pretable patterns. Rather, they show many intuitively desirable properties such as compositionally, increas- ing invariance and class discrimination as we ascend the layers. We also demonstrated through a series of occlusion experiments that the model, while trained for classification, is highly sensitive to local structure in the image and is not just using broad scene con- text. The experiments suggest that the feature rep- resentations produced by deep convnets may also be useful for more challenging tasks such as localization, segmentation and detection. The occlusion techniques also allowed us to probe how the model may be im- plicitly establishing correspondence with stable object parts in the upper feature layers, when determining its prediction. Finally, we showed how the ImageNet trained model can generalize well to other datasets. For Caltech-101 and Caltech-256, the datasets are sim- ilar enough that we can beat the best reported results, in the latter case by a significant margin. This re- sult brings into question to utility of benchmarks with small (i.e. *<* 104) training sets. Our convnet model generalized less well to the PASCAL data, perhaps suffering from dataset bias ([Torralba & Efros](#bookmark20), [2011](#bookmark20)), although it was still within 3*.*2% of the best reported result, despite no tuning for the task. For example, our performance might improve if a different loss function was used that permitted multiple objects per image.



 

#                     ![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image074.jpg)                        ![img]()Layer 1

 

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![img](file:////Users/haohao/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image086.png) |





 

 

 Layer 2 

 

 

 

 

 

 

 

 

 

 

 

 Layer 3 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 Layer 4                              Layer 5 

*Figure 8.* Visualization of features in a fully trained model. For layers 2-5 we show the top 9 activations in a random subset of feature maps across the validation data, projected down to pixel space using our deconvolutional network approach. Our reconstructions are *not* samples from the model: they are reconstructed patterns from the validation set that cause high activations in a given feature map. For each feature map we also show the corresponding image patches. Note:

(i) the the strong grouping within each feature map, (ii) greater invariance at higher layers and (iii) exaggeration of discriminative parts of the image, e.g. eyes and noses of dogs (layer 4, row 1, cols 1). Best viewed in electronic form.



## References

Bengio, Y., Lamblin, P., Popovici, D., and Larochelle,

H. Greedy layer-wise training of deep networks. In

*NIPS*, pp. 153–160, 2007.

Berkes, P. and Wiskott, L. On the analysis and in- terpretation of inhomogeneous quadratic forms as receptive fields. *Neural Computation*, 2006.

Bo, L., Ren, X., and Fox, D. Multipath sparse coding using hierarchical matching pursuit. In *CVPR*, 2013.

Ciresan, D. C., Meier, J., and Schmidhuber, J. Multi- column deep neural networks for image classifica- tion. In *CVPR*, 2012.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. ImageNet: A Large-Scale Hierarchical Image Database. In *CVPR09*, 2009.

Erhan, D., Bengio, Y., Courville, A., and Vincent, P. Visualizing higher-layer features of a deep network. In *Technical* *report,* *University of Montreal*, 2009.

Fei-fei, L., Fergus, R., and Perona, P. One-shot learn- ing of object categories. *IEEE Trans. PAMI*, 2006.

Griffin, G., Holub, A., and Perona, P. The caltech 256. In *Caltech Technical Report*, 2006.

Hinton, G. E., Osindero, S., and The, Y. A fast learn- ing algorithm for deep belief nets. *Neural Computa- tion*, 18:1527–1554, 2006.

Hinton, G.E., Srivastave, N., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. R. Improv- ing neural networks by preventing co-adaptation of feature detectors. arXiv:1207.0580, 2012.

Jarrett, K., Kavukcuoglu, K., Ranzato, M., and Le- Cun, Y. What is the best multi-stage architecture for object recognition? In *ICCV*, 2009.

Jianchao, Y., Kai, Y., Yihong, G., and Thomas, H. Linear spatial pyramid matching using sparse cod- ing for image classification. In *CVPR*, 2009.

Krizhevsky, A., Sutskever, I., and Hinton, G.E. Im- agenet classification with deep convolutional neural networks. In *NIPS*, 2012.

Le, Q. V., Ngiam, J., Chen, Z., Chia, D., Koh, P., and Ng, A. Y. Tiled convolutional neural networks. In *NIPS*, 2010.

LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., and Jackel, L. D. Backpropagation applied to handwritten zip code recognition. *Neural Comput.*, 1(4):541–551, 1989.





Sande, K., Uijlings, J., Snoek, C., and Smeulders, A. Hybrid coding for selective search. In *PASCAL* *VOC Classification Challenge 2012*, 2012.

Sohn, K., Jung, D., Lee, H., and Hero III, A. Effi- cient learning of sparse, distributed, convolutional feature representations for object recognition. In *ICCV*, 2011.

Torralba, A. and Efros, A. A. Unbiased look at dataset bias. In *CVPR*, 2011.

Vincent, P., Larochelle, H., Bengio, Y., and Manzagol,

P. A. Extracting and composing robust features with denoising autoencoders. In *ICML*, pp. 1096– 1103, 2008.

Yan, S., Dong, J., Chen, Q., Song, Z., Pan, Y., Xia, W., Huang, Z., Hua, Y., and Shen, S. Generalized hierarchical matching for sub-category aware object classification. In *PASCAL* *VOC Classification Chal- lenge 2012*, 2012.

Zeiler, M., Taylor, G., and Fergus, R. Adaptive decon- volutional networks for mid and high level feature learning. In *ICCV*, 2011.