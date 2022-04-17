# Deeplab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully connected CRFs


**Abstract**-In this work we address the task of semantic image segmentation with Deep Learning and make three main contributions that are experimentally shown to have substantial practical merit. First, we highlight convolution with upsampled filters, or 'atrous convolution', as a powerful tool in dense prediction tasks. Atrous convolution allows us to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. It also allows us to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation. Second, we propose atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales. Third, we improve the localization of object boundaries by combining methods from DCNNs and probabilistic graphical models. The commonly deployed combination of max-pooling and downsampling in DCNNs achieves invariance but has a toll on localization accuracy. We overcome this by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance. Our proposed ”DeepLab“ system sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 79.7% mIOU in the test set, and advances the results on three other datasets: PASCAL-Context, PASCAL-Person-Part, and Cityscapes. All of our code is made publicly available online.


**摘要**  本文展示了基于深度学习的图像语义分割工作，主要有三个经实验验证有实际应用价值的贡献。第一，本文强调了有上采样滤波器的卷积核，或者说‘空洞卷积’，是密集预测任务中的一个有力工具；空洞卷积可以使我们显示的控制DCNN计算得到的特征响应的分辨率；也可以有效地增大滤波器的视野以利用更大范围的上下文，而不需要增加参数量或计算量。第二，本文提出了空洞空间金字塔池化（ASPP），可以在多维度鲁棒的分割目标。ASPP利用了含有多个采样率和有效视野的滤波器的卷积特征层，所以能在多个尺度下捕捉目标和图像上下文。第三，本文借鉴结合了CDNNs和概率图模型的思路，提升了目标边缘的定位。在DCNNs中通常是组合使用max-pooling和downsample，这具有不变性，但是牺牲了定位准确性。本文将DCNN最后层响应与一个全连接条件随机场（CRF）相结合，在数量和质量两方面提升了定位准确性，解决了该问题。本文提出的Deepab系统在PASCAL VOC-2012语义分割任务中取得了目前最好的结果，在测试集中达到了79.7%mIOU的成果，并在其他三个数据集上推进了结果：PASCAL-Context, PASCAL-Person-Part, and Cityscapes。所有代码都已开源。


**Index Terms**—Convolutional Neural Networks, Semantic Segmentation, Atrous Convolution, Conditional Random Fields。


**关键字**---卷积神经网络，语义分割，空洞卷积，条件随机场（CRF）


## 1 INTRODUCTION 引言

Deep Convolutional Neural Networks (DCNNs) [1] have pushed the performance of computer vision systems to soaring heights on a broad array of high-level problems, including image classification [2], [3], [4], [5], [6] and object detection [7], [8], [9], [10], [11], [12], where DCNNs trained in an end-to-end manner have delivered strikingly better results than systems relying on hand-crafted features. Essential to this success is the built-in invariance of DCNNs to local image transformations, which allows them to learn increasingly abstract data representations [13]. This invariance is clearly desirable for classification tasks, but can hamper dense prediction tasks such as semantic segmentation, where abstraction of spatial information is undesired.

深度卷积神经网络（DCNNs）使得计算机视觉在许多高阶问题中取得了非常好的性能，包括图像分类问题，目标检测问题，其中DCNNs使用端到端方式训练模型，比手动设计的系统取得了印象深刻的好结果。成功的关键是DCNNs对局部图像变换的内在不变性，使得DCNNs可以学习到越来越抽象的数据表示。这种对局部图像变换的内在不变性非常有适合于图像分类任务，但不利于密集预测任务，例如语义分割，其中空间信息的抽象是不利的。


In particular we consider three challenges in the application of DCNNs to semantic image segmentation: (1) reduced feature resolution, (2) existence of objects at multiple scales, and (3) reduced localization accuracy due to DCNN invariance. Next, we discuss these challenges and our approach to overcome them in our proposed DeepLab system.


尤其是考虑到应用DCNNs进行语义分割时的三个挑战：（1）减少的特征分辨率，（2）目标对象存在于多个尺度，（3）因DCNN不变性而减小的定位准确性。此外，本文讨论了上述挑战和本文提出的Deeplab系统中克服上述问题的方法。


The first challenge is caused by the repeated combination of max-pooling and downsampling (‘striding’) performed at consecutive layers of DCNNs originally designed for image classification [2], [4], [5]. This results in feature maps with significantly reduced spatial resolution when the DCNN is employed in a fully convolutional fashion [14]. In order to overcome this hurdle and efficiently produce denser feature maps, we remove the downsampling operator from the last few max pooling layers of DCNNs and instead upsample the filters in subsequent convolutional layers, resulting in feature maps computed at a higher sampling rate. Filter upsampling amounts to inserting holes (‘trous’ in French) between nonzero filter taps. This technique has a long history in signal processing, originally developed for the efficient computation of the undecimated wavelet transform in a scheme also known as “algorithme `atrous” [15]. We use the term atrous convolution as a shorthand for convolution with upsampled filters. Various flavors of this idea have been used before in the context of DCNNs by [3], [6], [16]. In practice, we recover full resolution feature maps by a combination of atrous convolution, which computes feature maps more densely, followed by simple bilinear interpolation of the feature responses to the original image size. This scheme offers a simple yet powerful alternative to using deconvolutional layers [13], [14] in dense prediction tasks. Compared to regular convolution with larger filters, atrous convolution allows us to effectively enlarge the field of view of filters without increasing the number of parameters or the amount of computation.

第一个挑战是由于在为分类任务设计的DCNNs的连续卷积层中重复组合maxpooling与downsample（stride）操作导致的。当DCNNs是全卷积形式时，这会导致特征图的空间分辨率不断减小。为了克服这一障碍及高效生成更密集的特征图，本文移除了DCNNs最后几个最大池化层（maxpooling layer）中的下采样算子（downsampleoperator），在后续的卷积层中对滤波器进行上采样，使得特征层以较高的采样率被计算。滤波器上采样，即在滤波器的非零值中插入空洞。这种技术在信号处理中有很长的历史，最开始提出时是为了高效计算undecimated小波变换，称为空洞算法。本文称采用这种上采样滤波器的卷积为空洞卷积。这种算法的不同变体在DCNNs之前已经被应用了。本文恢复了全分辨率特征图，方法是使用空洞卷积的组合，然后进行简单的双线性插值，达到原始图像分辨率大小；空洞卷积能够更密集地计算特征图。与使用大滤波核地常规卷积相比，空洞卷积能够有效增加滤波器地视野，而不会增加参数量和计算量。


The second challenge is caused by the existence of objects at multiple scales. A standard way to deal with this is to present to the DCNN rescaled versions of the same image and then aggregate the feature or score maps [6], [17], [18]. We show that this approach indeed increases the performance of our system, but comes at the cost of computing feature responses at all DCNN layers for multiple scaled versions of the input image. Instead, motivated by spatial pyramid pooling [19], [20], we propose a computationally efficient scheme of resampling a given feature layer at multiple rates prior to convolution. This amounts to probing the original image with multiple filters that have complementary effective fields of view, thus capturing objects as well as useful image context at multiple scales. Rather than actually resampling features, we efficiently implement this mapping using multiple parallel atrous convolutional layers with different sampling rates; we call the proposed technique “atrous spatial pyramid pooling” (ASPP).

第二个挑战是目标在多尺度上存在导致的。此问题的一个标准处理方法是向DCNNs输入变换不同尺寸的同一张照片，然后将特征图或者分数图汇集起来。本文证明了这种方法确实提高了我们系统的性能，但其代价是要计算同一张图片多个变换尺寸的输入DCNNs中的所有特征相应。然而，受到空间金字塔池化启发，本文提出了对一个给定特征层在多个比率上重采样的高效计算方法。本文使用有效视野互补的多个滤波器处理原始图像，因此在多个尺度上不仅捕获了目标，而且捕获了有用的图像上下文信息。本文将多个具有不同采样率的空洞卷积层并联起来，以此高效实现了这种映射，而非真的重采样特征；本文称这种技术为孔洞空间金字塔赤化（ASPP）。


The third challenge relates to the fact that an object-centric classifier requires invariance to spatial transformations, inherently limiting the spatial accuracy of a DCNN. One way to mitigate this problem is to use skip-layers to extract “hyper-column” features from multiple network layers when computing the final segmentation result [14], [21]. Our work explores an alternative approach which we show to be highly effective. In particular, we boost our model’s ability to capture fine details by employing a fully connected Conditional Random Field (CRF) [22]. CRFs have been broadly used in semantic segmentation to combine class scores computed by multi-way classifiers with the low level information captured by the local interactions of pixels and edges [23], [24] or superpixels [25]. Even though works of increased sophistication have been proposed to model the hierarchical dependency [26], [27], [28] and/or high order dependencies of segments [29], [30], [31], [32], [33], we use the fully connected pairwise CRF proposed by [22] for its efficient computation, and ability to capture fine edge details while also catering for long range dependencies. That model was shown in [22] to improve the performance of a boosting-based pixel-level classifier. In this work, we demonstrate that it leads to state-of-the-art results when coupled with a DCNN-based pixel-level classifier.

第三个挑战和以下事实相关：即以目标为中心的分类器要求具有对空间变换的不变性，其内在上限制了DCNN的空间精确度。一个解决方法是，使用跳跃层从多个网络层中提取 高级 （hyper-column）特征，然后计算最终分割结果。我们探索了另外一种非常有效的方法。特别地，本文采用一种全连接的条件随机场（CRF）来提升模型捕获精细细节的能力。CRFs在语义分割中有广泛应用，结合了多路分类器计算出的类别得分和低层信息，这些低层信息是从像素与边缘，或超像素的局部反应捕捉到的。即使提出了越来越复杂的模型来对层次依赖关系和片段的高阶依赖关系进行建模，但我们还是使用[22]提出的全连接成对CRF，因为计算效率高，而且能够捕获精细边缘细节，同时照顾到长程依赖关系。模型在[22]中改进了一种基于boosting的像素级分类器。本文中，我们证明了，当与基于DCNN的像素级分类器结合时，可以得到目前最好的结果。


A high-level illustration of the proposed DeepLab model is shown in Fig.1. A deep convolutional neural network (VGG-16 [4] or ResNet-101 [11] in this work) trained in the task of image classification is re-purposed to the task of semantic segmentation by (1) transforming all the fully connected layers to convolutional layers (i.e., fully convolutional network [14]) and (2) increasing feature resolution through atrous convolutional layers, allowing us to compute feature responses every 8 pixels instead of every 32 pixels in the original network. We then employ bi-linear interpolation to upsample by a factor of 8 the score map to reach the original image resolution, yielding the input to a fully connected CRF [22] that refines the segmentation results.
图1为本文提出的DeepLab模型概要。用于图像分类任务的DCNNs（本文中是VGG16或ResNet101）以下修改后用于图像语义分割任务：将所有全连接层修改为卷积层；通过空洞卷积方法提高特征分辨率，可以没8像素计算一个特征而不是原来的32像素。本文然后采用双线性插值来将特征图上采样到原始图像分辨率。结果输入到全连接CRF中，得到精炼的分割结果。


From a practical standpoint, the three main advantages of our DeepLab system are: (1) Speed: by virtue of atrous convolution, our dense DCNN operates at 8 FPS on an NVidia Titan X GPU, while Mean Field Inference for the
fully-connected CRF requires 0.5 secs on a CPU. (2) Accuracy: we obtain state-of-art results on several challenging datasets, including the PASCAL VOC 2012 semantic segmentation benchmark [34], PASCAL-Context [35], PASCALPerson-Part [36], and Cityscapes [37]. (3) Simplicity: our system is composed of a cascade of two very well-established modules, DCNNs and CRFs.
从实际观点看，本文的DeepLab系统主要有三个优势:(1)速度:利用孔洞卷积的优势，本文的密集DCNNs在NVidia X GPU上取得了8FPS的速率，而在cpu上全连接CRF的平均推理时间需要0.5秒。(2)准确性：本文模型在几个挑战数据集上取得当前最好的结果，包括PASCAL VOC2012语义分割竞赛，PASCAL-Context，PASCAL Person-Part, and Cityscapes。（3）简单性：本文的系统是由两个设计良好的模型，DCNNs和CRFs，级联而成。



The updated DeepLab system we present in this paper features several improvements compared to its first version
reported in our original conference publication [38]. Our new version can better segment objects at multiple scales, via either multi-scale input processing [17], [39], [40] or the proposed ASPP. We have built a residual net variant of DeepLab by adapting the state-of-art ResNet [11] image classification DCNN, achieving better semantic segmentation performance compared to our original model based on VGG-16 [4]. Finally, we present a more comprehensive experimental evaluation of multiple model variants and report state-of-art results not only on the PASCAL VOC 2012 benchmark but also on other challenging tasks. We have implemented the proposed methods by extending the Caffe framework [41]. We share our code and models at a companion web site http://liangchiehchen.com/projects/ DeepLab.html.


本文中升级后的DeepLab系统与第一代版本相比有几个提升。新版本能够更好地在多尺度上分割目标，不论是通过多尺度的输入处理或者是本文提出的ASPP模块。本文改造了RenNet图像分类的DCNN网络形成了DeepLab的残差网络变体，与第一版基于VGG16的DeepLabV1相比，达到了更好的语义分割性能。最后，本文对多个模型变体进行了更为全面的实验评估，不仅在 PASCAL VOC 2012 基准测试中，也在其他挑战任务中取得了最好的成绩。本文使用Caffe框架实现了模型。代码和模型共享于以下网址：http://liangchiehchen.com/projects/ DeepLab.html。

## 2 Related Work 相关工作

Most of the successful semantic segmentation systems developed in the previous decade relied on hand-crafted features combined with flat classifiers, such as Boosting [24],[42], Random Forests [43], or Support Vector Machines [44]. Substantial improvements have been achieved by incorporating richer information from context [45] and structured prediction techniques [22], [26], [27], [46], but the performance of these systems has always been compromised by the limited expressive power of the features. Over the past few years the breakthroughs of Deep Learning in image classification were quickly transferred to the semantic segmentation task. Since this task involves both segmentation and classification, a central question is how to combine the two tasks.
多数过去十年研提出的成功的语义分割系统都依赖于手动设计的特征，并与分类器结合，例如Boosting、随机森林、支持向量机等。大量的改进提升是通过将更丰富的上下文信息和结构化预测相结合而达到的，但是这些系统的性能被特征表达能力有限这一特点制约。过去几年深度学习在图像分类中取得的突破，很快迁移到图像语义分割任务中。因为语义分割任务与图像的分类和分割两个任务都相关，所以一个核心问题是如何把这两个任务结合起来。




The first family of DCNN-based systems for semantic segmentation typically employs a cascade of bottom-up image segmentation, followed by DCNN-based region classification. For instance the bounding box proposals and masked regions delivered by [47], [48] are used in [7] and [49] as inputs to a DCNN to incorporate shape information into the classification process. Similarly, the authors of [50] rely on a superpixel representation. Even though these approaches can benefit from the sharp boundaries delivered by a good segmentation, they also cannot recover from any of its errors
第一类基于DCNN的语义分割系统主要使用级联方法，



The second family of works relies on using convolutionally computed DCNN features for dense image labeling, and couples them with segmentations that are obtained independently. Among the first have been [39] who apply DCNNs at multiple image resolutions and then employ a segmentation tree to smooth the prediction results. More recently, [21] propose to use skip layers and concatenate the computed intermediate feature maps within the DCNNs for pixel classification. Further, [51] propose to pool the intermediate feature maps by region proposals. These works still employ segmentation algorithms that are decoupled from the DCNN classifier’s results, thus risking commitment to premature decisions.







The third family of works uses DCNNs to directly provide dense category-level pixel labels, which makes it possible to even discard segmentation altogether. The segmentation-free approaches of [14], [52] directly apply DCNNs to the whole image in a fully convolutional fashion, transforming the last fully connected layers of the DCNN into convolutional layers. In order to deal with the spatial localization issues outlined in the introduction, [14] upsample and concatenate the scores from intermediate feature maps, while [52] refine the prediction result from coarse to fine by propagating the coarse results to another DCNN. Our work builds on these works, and as described in the introduction extends them by exerting control on the feature resolution, introducing multi-scale pooling techniques and integrating the densely connected CRF of [22] on top of the DCNN. We show that this leads to significantly better segmentation results, especially along object boundaries. The combination of DCNN and CRF is of course not new but previous works only tried locally connected CRF models. Specifically, [53] use CRFs as a proposal mechanism for a DCNN-based reranking system, while [39] treat superpixels as nodes for a local pairwise CRF and use graph-cuts for discrete inference. As such their models were limited by errors in superpixel computations or ignored long-range dependencies. Our approach instead treats every pixel as a CRF node receiving unary potentials by the DCNN. Crucially, the Gaussian CRF potentials in the fully connected CRF model of [22] that we adopt can capture long-range dependencies and at the same time the model is amenable to fast mean field inference. We note that mean field inference had been extensively studied for traditional image segmentation tasks [54], [55], [56], but these older models were typically limited to shortrange connections. In independent work, [57] use a very similar densely connected CRF model to refine the results of DCNN for the problem of material classification. However, the DCNN module of [57] was only trained by sparse point supervision instead of dense supervision at every pixel.



Since the first version of this work was made publicly available [38], the area of semantic segmentation has progressed drastically. Multiple groups have made important advances, significantly raising the bar on the PASCAL VOC 2012 semantic segmentation benchmark, as reflected to the high level of activity in the benchmark’s leaderboard1 [17], [40], [58], [59], [60], [61], [62], [63]. Interestingly, most top performing methods have adopted one or both of the key ingredients of our DeepLab system: Atrous convolution for efficient dense feature extraction and refinement of the raw DCNN scores by means of a fully connected CRF. We outline below some of the most important and interesting advances.



End-to-end training for structured prediction has more recently been explored in several related works. While we employ the CRF as a post-processing method, [40], [59], [62], [64], [65] have successfully pursued joint learning of the DCNN and CRF. In particular, [59], [65] unroll the CRF mean-field inference steps to convert the whole system into an end-to-end trainable feed-forward network, while [62] approximates one iteration of the dense CRF mean field inference [22] by convolutional layers with learnable filters. Another fruitful direction pursued by [40], [66] is to learn the pairwise terms of a CRF via a DCNN, significantly improving performance at the cost of heavier computation. In a different direction, [63] replace the bilateral filtering module used in mean field inference with a faster domain transform module [67], improving the speed and lowering the memory requirements of the overall system, while [18], [68] combine semantic segmentation with edge detection.



Weaker supervision has been pursued in a number of papers, relaxing the assumption that pixel-level semantic annotations are available for the whole training set [58], [69], [70], [71], achieving significantly better results than weakly supervised pre-DCNN systems such as [72]. In another line of research, [49], [73] pursue instance segmentation, jointly tackling object detection and semantic segmentation.


What we call here atrous convolution was originally developed for the efficient computation of the undecimated wavelet transform in the “algorithme `a trous” scheme of [15]. We refer the interested reader to [74] for early references from the wavelet literature. Atrous convolution is also intimately related to the “noble identities” in multi-rate signal processing, which builds on the same interplay of input signal and filter sampling rates [75]. Atrous convolution is a term we first used in [6]. The same operation was later called dilated convolution by [76], a term they coined motivated by the fact that the operation corresponds to regular convolution with upsampled (or dilated in the terminology of [15]) filters. Various authors have used the same operation before for denser feature extraction in DCNNs [3], [6], [16]. Beyond mere resolution enhancement, atrous convolution allows us to enlarge the field of view of filters to incorporate larger context, which we have shown in [38] to be beneficial. This approach has been pursued further by [76], who employ a series of atrous convolutional layers with increasing rates to aggregate multiscale context. The atrous spatial pyramid pooling scheme proposed here to capture multiscale objects and context also employs multiple atrous convolutional layers with different sampling rates, which we however lay out in parallel instead of in serial. Interestingly, the atrous convolution technique has also been adopted for a broader set of tasks, such as object detection [12], [77], instance level segmentation [78], visual question answering [79], and optical flow [80].

We also show that, as expected, integrating into DeepLab more advanced image classification DCNNs such as the residual net of [11] leads to better results. This has also been observed independently by [81].



## 3 METHODS 方法

### 3.1 Atrous Convolution for Dense Feature Extraction and Field-of-View Enlargement


The use of DCNNs for semantic segmentation, or other dense prediction tasks, has been shown to be simply and successfully addressed by deploying DCNNs in a fully convolutional fashion [3], [14]. However, the repeated combination of max-pooling and striding at consecutive layers of these networks reduces significantly the spatial resolution of the resulting feature maps, typically by a factor of 32 across each direction in recent DCNNs. A partial remedy is to use ‘deconvolutional’ layers as in [14], which however requires additional memory and time.

We advocate instead the use of atrous convolution, originally developed for the efficient computation of the undecimated wavelet transform in the “algorithme `a trous” scheme of [15] and used before in the DCNN context by [3], [6], [16]. This algorithm allows us to compute the responses of any layer at any desirable resolution. It can be applied post-hoc, once a network has been trained, but can also be seamlessly integrated with training.




