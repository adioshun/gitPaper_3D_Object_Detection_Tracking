| 논문명 | PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation |
| --- | --- |
| 저자\(소속\) | Charles R. Qi \(Stanford University\)|
| 학회/년도 | Dec 2016 ~ Apr 2017, CVPR 2017,  [논문](https://arxiv.org/abs/1612.00593) |
| 키워드 |Charles2016, Classification, Segmentation    |
| 데이터셋(센서)/모델| ModelNet40,  ShapeNet part data, Stanford 3D semantic parsing data|
|관련연구|이후연구 : Pointnet++|
| 참고 | [ppt](https://www.facebook.com/thinking.factory/posts/1408857052524274), [홈페이지](http://stanford.edu/~rqi/pointnet/),[CVPR2017](https://www.youtube.com/watch?v=Cge-hot0Oc0), [Youtube](https://youtu.be/8CenT_4HWyY?t=1h16m39s), [ppt](http://3ddl.stanford.edu/CVPR17_Tutorial_PointCloud.pdf), [한글설명](http://daddynkidsmakers.blogspot.com/2017/07/3-pointnet.html)  |
| 코드 | [TF](https://github.com/charlesq34/pointnet), [pyTorch](https://github.com/fxia22/pointnet.pytorch)|

> 같은 이름의 다른 논문 :  PointNet: A 3D Convolutional Neural Network for real-time object class recognition, A. Garcia-Garcia, [링크](http://ieeexplore.ieee.org/document/7727386/authors)

PointNet : [TF_Official](https://github.com/charlesq34/pointnet), [pyTorch](https://github.com/fxia22/pointnet.pytorch), [pyTorch-Open3D](https://github.com/intel-isl/Open3D-PointNet)



도커 : [PointNet-pyTorch](https://hub.docker.com/r/dkwatson/pointnet-pytorch1)


포인트 클라우드를 직접 사용함으로써 **permutation invariance**한 입력 정보의 속성을 고려 하였다. 제안 방식은 **object classification, part segmentation, to scene semantic parsing**가 가능 하다. `Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images. This, however, renders data unnecessarily voluminous and causes issues. In this paper, we design a novel type of neural network that directly consumes point clouds, which well respects the permutation invariance of points in the input. Our network, named PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective. Empirically, it shows strong performance on par or even better than state of the art. Theoretically, we provide analysis towards understanding of what the network has learnt and why the network is robust with respect to input perturbation and corruption.`



```
# [페이스북정리글](https://www.facebook.com/groups/TensorFlowKR/permalink/508389389502124/)

PointNet : End-to-end learning for scattered, unordered point data, Unified framework for various tasks

> Point Cloud : 가장 직관적이고 정보를 잘 표현할 수 있으면서도 다른 방법에서 변환하기도 쉽고 역으로 돌아가기도 쉬우며 얻기도 편하므로 이를 많이 사용한다.

Two Challenges
- Unordered point set as input : data의 order에 invariant해야한다는 점 
- Invariance under geometric transformations : point clouds에 geometric transformation을 가한다고 물체가 다른 것으로 분류되어서도 안된다는 점 

본 논문은
- 첫번쨰(permutation invariance)는 symmetric function을 도입하여 해결
- 두번째는 transformer network를 하나 모듈로 붙여서 해결하였다.

PointNet은 max pooling을 기준으로 앞부분의 local feature단과 뒷부분의 global feature단을 보는 것으로 나눌 수 있는데, 논문에서는 critical point로 불리는 global feature에 영향을 주는 point set은 매우 적고 주요 경계마다만 있고 대다수의 point들은 영향을 주지 않기 떄문에 전에 point clouds에서 50%까지 data loss가 있더라도 전혀 성능에 문제가 발생하지 않는다. \(robustness to missing data\)


- Qi et al.  propose a Multilayer Perceptron(MLP) architecture 
	-  that extracts a global feature vector from a 3D point cloud of $$1m^3$$ physical size 
	-  and processes each point using the extracted feature vector and additional **point level** transformations. 

- Their method operates at the point level and thus inherently provides a fine-grained segmentation. 

- It works well for indoor semantic scene understanding,although there is no evidence that it scales to larger input dimensions without additional training or adaptation required. 
```



# PointNet


## 1. Introduction

기존 연구 방법 Due to Point cloud's irregular format, most researchers transform such data to

* regular 3D voxel grids or 
* collections of images

기존 연구 문제점 :  This data representation transformation, however, renders the resulting data unnecessarily voluminous — while also introducing quantization artifacts that can obscure natural invariances of the data


제안 방식 
- 입력 
    - three coordinates (x, y, z)
    - normals 
    - local or global features
- 출력 
    - classifiaction : label
    - Segmentation : 각 point별 label 
    

네트워크 특징 

- a single symmetric function, max pooling. 

    - Effectively the network learns a set of optimization functions/criteria that select interesting or informative points of the point cloud and encode the reason for their selection. 

- The final fully connected layers of the network aggregate these learnt optimal values into the global descriptor for the entire shape as mentioned above (shape classification) or are used to predict per point labels (shape segmentation).




## 2. Related Work

### 2.1 Point Cloud Features

> Point Cloud 특징들은 `수작업`으로 만든것들이다. 이전 CV 방식 처럼

Most existing features for point cloud are `handcrafted` towards specific tasks.

Point features often encode certain statistical properties of points and are designed to be invariant to certain transformations, 


분류 1
- intrinsic \[2, 24, 3\] 
- extrinsic \[20, 19, 14, 10, 5\].

분류 2
- local features
- global features

For a specific task, it is not trivial to find the optimal feature combination.

### 2.2 Deep Learning on 3D Data

3D data has multiple popular representations, leading to various approaches for learning.

#### A. Volumetric CNNs

- \[28, 17, 18\] are the pioneers applying 3D convolutional neural networks on voxelized shapes.

> ShpaeNet, VoxNet, Vol/Multi-View CNNs

제약 : sparsity problem, 계산 부하 `However, volumetric representation is constrained by its resolution due to data sparsity and computation cost of 3D convolution.`

sparsity 문제 해결 논문 
- FPNN [13]
- Vote3D [26]
- however, their operations are still on sparse volumes, it’s challenging for them to process very large point clouds.

```
[28] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1912–1920, 2015.
[17] D. Maturana and S. Scherer. Voxnet: A 3d convolutional neural network for real-time object recognition. In IEEE/RSJ International Conference on Intelligent Robots and Systems, September 2015
[18] C. R. Qi, H. Su, M. Nießner, A. Dai, M. Yan, and L. Guibas. Volumetric and multi-view cnns for object classification on 3d data. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2016. 
[13] Y. Li, S. Pirk, H. Su, C. R. Qi, and L. J. Guibas. Fpnn: Field probing neural networks for 3d data. arXiv preprint arXiv:1605.06240, 2016.
[26] D. Z. Wang and I. Posner. Voting for voting in online point cloud object detection. Proceedings of the Robotics: Science and Systems, Rome, Italy, 1317, 2015
```

#### B. Multiview CNNs

> 3D Point Cloud를 2D 이미지로 맵핑하고 2D CNN을 접목하는 방법, 성능이 잘 나옴

- \[23, 18\] have tried to render 3D point cloud or shapes into 2D images and then apply 2D conv nets to classify them.

- With well engineered image CNNs, this line of methods have achieved dominating performance on shape classification and retrieval tasks \[21\].

- However, it’s nontrivial to extend them to scene understanding or other 3D tasks such as point classification and shape completion.

```
[23] H. Su, S. Maji, E. Kalogerakis, and E. G. Learned-Miller. Multi-view convolutional neural networks for 3d shape recognition. In Proc. ICCV, to appear, 2015
[18] C. R. Qi, H. Su, M. Nießner, A. Dai, M. Yan, and L. Guibas. Volumetric and multi-view cnns for object classification on 3d data. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2016
[21] M. Savva, F. Yu, H. Su, M. Aono, B. Chen, D. Cohen-Or, W. Deng, H. Su, S. Bai, X. Bai, et al. Shrec16 track largescale 3d shape retrieval from shapenet core55.
```

#### D. Spectral CNNs

- Some latest works \[4, 16\] use spectral CNNs on meshes.

- However, these methods are currently constrained on manifold meshes such as organic objects and it’s not obvious how to extend them to non-isometric shapes such as furniture.

```
[4] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun. Spectral networks and locally connected networks on graphs. arXiv preprint arXiv:1312.6203, 2013
[16] J. Masci, D. Boscaini, M. Bronstein, and P. Vandergheynst. Geodesic convolutional neural networks on riemannian manifolds. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 37–45, 2015.
```

#### E. Feature-based DNNs

- \[6, 8\]firstly convert the 3D data into a vector, by extracting traditional shape features and then use a fully connected net to classify the shape.

- 제약 : We think they are constrained by the representation power of the features extracted.

> 특징 추출단계에서 표현력 제약 발생 가능

```
[6] Y. Fang, J. Xie, G. Dai, M. Wang, F. Zhu, T. Xu, and E. Wong. 3d deep shape descriptor. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2319–2328, 2015. 
[8] K. Guo, D. Zou, and X. Chen. 3d mesh labeling via deep convolutional neural networks. ACM Transactions on Graphics (TOG), 35(1):3, 2015.
```

### 2.3 Deep Learning on Unordered Sets

* 구조적 관점에서 보면 포인트 클라우드는 **Unordered**이다. `From a data structure point of view, a point cloud is an unordered set of vectors.`

* 대화, 언어, 비디오, 3D\(Volumes\)들에 대하여서는 연구 되었지만, 포인트 클라우드에 대해서는 연구 되지 않았다. `While most works in deep learning focus on regular input representations like sequences (in speech and language processing), images and volumes (video or 3D data), not much work has been done in deep learning on point sets.`

> 저자는 3D를 `volumes 과 unordered로 나누어서 보고있음 `

* 그나마 최근 연구는 \[25\]이다. `One recent work from Oriol Vinyals et al [25] looks into this problem.`

    * They use a read-process-write network with attention mechanism to consume unordered input sets and show that their network has the ability to sort numbers.

    - 그러나 이 연구 역시 NLP에 초점을 두고 있어서 Geo정보처리는 안한다. `However, since their work focuses on generic sets and NLP applications, there lacks the role of geometry in the sets.`

```
[25] O. Vinyals, S. Bengio, and M. Kudlur. Order matters: Sequence to sequence for sets. arXiv preprint arXiv:1511.06391, 2015.
```


## 3. Problem Statement


We design a deep learning framework that directly consumes unordered point sets as inputs. 

A point cloud is represented as a set of 3D points $$\{P_i \mid i = 1, ..., n \} 
- each point $$P_i$$ is a vector of its $$(x, y, z)$$ coordinate plus extra feature channels such as color, normal etc. 

For simplicity and clarity, unless otherwise noted, we only use the (x, y, z) coordinate as our point’s channels.

### 3.1 classification

입력 : For the object classification task, the **input **point cloud is 
    - 물체에서 직접 샘플링됨 `either directly sampled from a shape `
    - 공간에서 사전 분활됨 `or pre-segmented from a scene point cloud. `

출력 : 후보 분류에 대한 확률 정보를 제공한다. `Our proposed deep network outputs k scores for all the k candidate classes. `

### 3.2 semantic segmentation

입력 : For semantic segmentation, the **input** can be 
    - a single object for part region segmentation, 
    - or a sub-volume from a 3D scene for object region segmentation. 

출력 : Our model will output `n × m` scores for each of the `n` points and each of the `m` semantic subcategories.


## 4. Deep Learning on Point Sets


The architecture of our network (Sec 4.2) is inspired by the properties of point sets in $$\Re^n$$ (Sec 4.1).



### 4.1 Properties of Point Sets in $$\Re^n$$ 

Our input is a subset of points from an **Euclidean space**.

3가지 주요 특징 `It has three main properties:`

#### A. Unordered  (순서 관련 없음)

- Unlike pixel arrays in images or voxel arrays in volumetric grids, point cloud is a set of points without specific order. 

- In other words, a network that consumes $$N$$ 3D point sets needs to be invariant to $$N!$$ permutations of the input set in data feeding order.

#### B. Interaction among points. (포인트간 상호성)

- 포인트 들은 거리정보를 가지고 서로 떨어져 있다. `The points are from a space with a distance metric. `
	- It means that points are not isolated, and neighboring points form a meaningful subset. 
	- Therefore, the model needs to be able to capture local structures from nearby points, and the combinatorial interactions among local structures.


#### C. Invariance under transformations (변화에 불변성)

- As a geometric object, the learned representation of the point set should be invariant to certain transformations. 

- For example, **rotating** and **translating** points all together should 
    - not modify the global point cloud category 
    - nor the segmentation of the points.

### 4.2 PointNet Architecture

![](https://i.imgur.com/LZiDf16.png)
```
[Figure 2. PointNet Architecture.] 
- 분류 네트워크는 The classification network takes n points as input, applies input and feature transformations, and then aggregates point features by max pooling. 
    - The output is classification scores for k classes. 
- 분할 네트워크 The segmentation network is an extension to the classification net. 
    - It concatenates global and local features and outputs per point scores. 
    - “mlp” stands for multi-layer perceptron, numbers in bracket are layer sizes. 
    - Batchnorm is used for all layers with ReLU. 
    - Dropout layers are used for the last mlp in classification net.
```

Our full network architecture is visualized in Fig 2, where the classification network and the segmentation network share a great portion of structures. Please read the caption of Fig 2 for the pipeline.

Our network has three key modules: 
1. The max pooling layer as a **symmetric function** to aggregate information from all the points, 
2. a local and global information combination structure, 
3. and two joint alignment networks that align both input points and point features.


#### A. Symmetry Function for Unordered Input

입력 순서(`=permutation`)에 영향받지 않는 모델 만드는 3가지 방법 `In order to make a model invariant to input permutation, three strategies exist:` 
1. sort input into a canonical order
2. treat the input as a sequence to train an RNN, but augment the training data by all kinds of permutations;
3. use a simple symmetric function to aggregate the information from each point. 

##### 가. symmetric function (3번)

이 함수는 입력으로 n벡터를 받아서 출력으로 새 백터를 출력한다. 새 백터는 **입력 순서**에 강건하게 된다. `Here, a symmetric function takes `n` vectors as input and outputs a new vector that is invariant to the input order.` 

- For example, `+` and `∗` operators are **symmetric binary functions**.

##### 나. sorting (1번)

- 정렬이 간단한 방법 같지만 고차원에서는 정렬방법은 없다. `While sorting sounds like a simple solution, in high dimensional space there in fact does not exist an ordering that is stable w.r.t. point perturbations in the general sense. This can be easily shown by contradiction. `

- 비록 있다고 하더라도 고차원 공간과 1d real line간의 bijection map을 정의한것이다. `If such an ordering strategy exists, it defines a bijection map between a high-dimensional space and a 1d real line. `

It is not hard to see, to require an ordering to be stable w.r.t point perturbations is equivalent to requiring that this map preserves spatial proximity as the dimension reduces, a task that cannot be achieved in the general case. 

Therefore,sorting does not fully resolve the ordering issue, and it’s hard for a network to learn a consistent mapping from input to output as the ordering issue persists. 

As shown in experiments (Fig 5), we find that applying a MLP directly on the sorted point set performs poorly, though slightly better than directly processing an unsorted input.

##### 다. RNN (2번)

The idea to use RNN considers the point set as a sequential signal and hopes that by training the RNN with randomly permuted sequences, the RNN will become invariant to input order. 

However in “Order Matters” [25]the authors have shown that order does matter and cannot be totally omitted. 

While RNN has relatively good robustness to input ordering for sequences with small length (dozens), it’s hard to scale to thousands of input elements, which is the common size for point sets. 

Empirically, we have also shown that model based on RNN does not perform as well as our proposed method (Fig 5).


##### 라. 제안 방식 

Our idea is to approximate a general function defined on a point set by applying a **symmetric function** on transformed elements in the set:

![](https://i.imgur.com/RxCawYT.png)

- we approximate `h` by a multi-layer perceptron network and `g` by a composition of a single variable function and a max pooling function. 

- This is found to work well by experiments. 

- Through a collection of `h`, we can learn a number of `f’s` to capture different properties of the set. 



### [ 입력 순서에 강건성을 가지는 3가지 방법 ]
![](https://i.imgur.com/aFUsYsC.png)
```
[Figure 5. Three approaches to achieve order invariance.] 
- Multilayer perceptron (MLP) applied on points consists of 5 hidden layers with neuron sizes 64,64,64,128,1024, all points share a single copy of MLP. 
- The MLP close to the output consists of two layers with sizes 512,256.
```

While our key module seems simple, it has interesting properties (see Sec 5.3) and can achieve strong performace (see Sec 5.1) in a few different applications. 

Due to the simplicity of our module, we are also able to provide theoretical analysis as in Sec 4.3.


#### B. Local and Global Information Aggregation

위 섹션의 산출물은 `[f1, . . . , fK]`형태의 백터이며 입력 데이터에 대한 **global signature**이다. 여기에 SVM이나 MLP분류기를 적용하여 분류작업을 수행 할수 있다. `The output from the above section forms a vector [f1, . . . , fK], which is a global signature of the input set. We can easily train a SVM or multi-layer perceptron classifier on the shape global features for classification.`

하지만, 세그멘테이션을 위해서는 Local + Global 정보가 필요하다.  `However, point segmentation requires a combination of local and global knowledge. We can achieve this by a simple yet highly effective manner.`

전역(Global) 특징 벡터 계산 후 이 정보를 다시 각 포인트별 특징과 합친후 입력으로 사용한다. 이후 추출된 각 포인트들은 전역/지역 특징을 모두 가지고 있다. ` Our solution can be seen in Fig 2 (Segmentation Network). After computing the global point cloud feature vector, we feed it back to per point features by concatenating the global feature with each of the point features. Then we extract new per point features based on the combined point features - this time the per point feature is aware of both the local and global information.`

이러한 조정후 With this modification our network is able to predict per point quantities that rely on both local geometry and global semantics. 

예를들어 Normal을 계산 하거나, 이웃 포인트간의 관계 정보를 축양 할수 있다. `For example we can accurately predict per-point normals (fig in supplementary), validating that the network is able to summarize information from the point’s local neighborhood. `

성능도 좋다. `In experiment session, we also show that our model can achieve state-of-the-art performance on shape part segmentation and scene segmentation.`

#### C. Joint Alignment Network

강제 변화의 영향을 받지 않아야 한다. `The semantic labeling of a point cloud has to be invariant if the point cloud undergoes certain geometric transformations, such as rigid transformation. We therefore expect that the learnt representation by our point set is invariant to these transformations.`

해결 방법은 특징 추출 전에 모든 입력을 **canonical space **에 정렬 하는것이다. ` A natural solution is to align all input set to a canonical space before feature extraction. `
- Jaderberg et al. [9] introduces the idea of spatial transformer to align 2D images through sampling and interpolation, achieved by a specifically tailored layer implemented on GPU.

```
[9] M. Jaderberg, K. Simonyan, A. Zisserman, et al. Spatial transformer networks. In NIPS 2015. 4
```

포인트 클라우드 입력 형태는 이미지 처리인 [9]보다 간단한게 목표를 달성 한다. `Our input form of point clouds allows us to achieve this goal in a much simpler way compared with [9]. We do not need to invent any new layers and no alias is introduced as in the image case. `
- **T-net**을 이용하여 **아핀 매트릭스**를 획득 한후 입력 포인트클라우드에 바로 적용한다. `We predict an affine transformation matrix by a mini-network (T-net in Fig 2) and directly apply this transformation to the coordinates of input points. `
- **T-net**은 특징 추출, 맥스풀링, FC등을 가지고 있는 네트워크 이다. `The mininetwork itself resembles the big network and is composed by basic modules of point independent feature extraction, max pooling and fully connected layers. More details about the T-net are in the supplementary.`

이 방법은 추후 특징 공간을 정렬 하는데도 활용 될수 있다. 하지만 **transformation matrix** 고차원이라 최적화가 어려워 본 논문에서는 은 **softmax training loss**에 **regularization**을 포함 시켰다. `This idea can be further extended to the alignment of feature space, as well. We can insert another alignment network on point features and predict a feature transformation matrix to align features from different input point clouds. However, transformation matrix in the feature space has much higher dimension than the spatial transform matrix, which greatly increases the difficulty of optimization. We therefore add a regularization term to our softmax training loss.`

We constrain the feature transformation matrix to be close to orthogonal matrix:
![](https://i.imgur.com/hniMboU.png)
- A가 **T-net**을 통해 구해진 특징 정렬 매트릭스 이다. `where A is the feature alignment matrix predicted by a mini-network. `

An orthogonal transformation will not lose information in the input, thus is desired. We find that by adding the regularization term, the optimization becomes more stable and our model achieves better performance.

> T-Net의 기본 아이디어는 [Spatial Transformer Network](https://arxiv.org/abs/1506.02025)에서 가져옴 [[출처]](https://github.com/charlesq34/pointnet/issues/31), [[코드위치]](https://github.com/charlesq34/pointnet/issues/74)

### 4.3. Theoretical Analysis

## 5. Experiment





---

# Supplementary 

## A. Overview 

This document provides additional quantitative results, technical details and more qualitative test examples to the main paper. 
- In Sec B we extend the robustness test to compare PointNet with VoxNet on incomplete input. 
- In Sec C we provide more details on neural network architectures, training parameters 
- and in Sec D we describe our detection pipeline in scenes. 
- Then Sec E illustrates more applications of PointNet, 
- while Sec F shows more analysis experiments. 
- Sec G provides a proof for our theory on PointNet. 
- At last, we show more visualization results in Sec H.


## C. Network Architecture and Training Details (Sec 5.1)

### C.1 PointNet Classification Network

기본 구조는 이미 설명 하였으므로 ** joint alignment/transformation**에 대하여 살펴 보겠다. `As the basic architecture is already illustrated in the main paper, here we provides more details on the joint alignment/transformation network and training parameters.`

- 첫번째 네트워크는 입력은 3 × 3 매트릭스 ` The first transformation network is a mini-PointNet that takes raw point cloud as input and regresses to a 3 × 3 matrix. `

- 구성은 It’s composed of a 
	- shared MLP(64, 128, 1024) network (with layer output sizes 64, 128, 1024) on each point, 
	- a max pooling across points and 
	- two fully connected layers with output sizes 512, 256. 

- 출력 매트릭스는 *identity matrix* 로 초기화 된다. 마지막 레이어를 제외 하고 모든 레이어는 ReLU와 BN이 적용 되어 있다. `The output matrix is initialized as an identity matrix. All layers, except the last one, include ReLU and batch normalization.`


- 두번째 네트워크는 첫번째와 같다. 단지 출력만 64x64 매트릭스 이다. `The second transformation network has the same architecture as the first one except that the output is a 64 × 64 matrix. The matrix is also initialized as an identity. A regularization loss (with weight 0.001) is added to the softmax classification loss to make the matrix close to orthogonal.`


사용된 파라미터는 다음과 같다. `We use dropout with keep ratio 0.7 on the last fully connected layer, whose output dimension 256, before class score prediction. The decay rate for batch normalization starts with 0.5 and is gradually increased to 0.99. We use adam optimizer with initial learning rate 0.001, momentum 0.9 and batch size 32. The learning rate is divided by 2 every 20 epochs. Training on ModelNet takes 3-6 hours to converge with TensorFlow and a GTX1080 GPU.`

### C.2 PointNet Segmentation Network

분활 네트워크는 분류 네트워크의 확장형이다. `The segmentation network is an extension to the classification PointNet.` 

지역 특징(두번째 변환 네트워크 결과물)과 전역 특징(맥스 풀링 결과물)을 각 포인트 별로 합쳐 진다. `Local point features (the output after the second transformation network) and global feature (output of the max pooling) are concatenated for each point.`

드랍아웃을 수행 되지 않으며, 각 파라미터는 분류기와 같다. ` No dropout is used for segmentation network. Training parameters are the same as the classification network.`

As to the task of shape part segmentation, we made a few modifications to the basic segmentation network architecture (Fig 2 in main paper) in order to achieve best performance, as illustrated in Fig 9. 

We add a one-hot vector indicating the class of the input and concatenate it with the max pooling layer’s output. We also increase neurons in some layers and add skip links to collect local point features in different layers and concatenate them to form point feature input to the segmentation network.

While [27] and [29] deal with each object category independently, due to the lack of training data for some categories (the total number of shapes for all the categories in the data set are shown in the first line), we train our PointNet across categories (but with one-hot vector input to indicate category). To allow fair comparison, when testing these two models, we only predict part labels for the given specific object category.

As to semantic segmentation task, we used the architecture as in Fig 2 in the main paper. It takes around six to twelve hours to train the model on ShapeNet part dataset and around half a day to train on the Stanford semantic parsing dataset.


## D. Details on Detection Pipeline (Sec 5.1)

We build a simple 3D object detection system based on the semantic segmentation results and our object classification PointNet.

전체에서 후보 물체를 찾기 위하여 connected component를 분할점수와 함께 사용 하였다. 전체에서 랜덤 포인트를 잡고 라벨값을 예측한후 BFS를 이용하여 같은 라벨을 가진 이웃 포인트들을 검색(탐색반경 0.2m)해 나아 간다. 클러스터링된 포인트가 200개 이사ㅇ이면 하나의 물체로 바운딩 박스 처리 된다. 각 후보 물체에 대하여 점수의 평균을 내어 detection score를 계산 한다. 너무 작은 물체는 제거 된다. `We use connected component with segmentation scores to get object proposals in scenes. Starting from a random point in the scene, we find its predicted label and use BFS to search nearby points with the same label, with a search radius of 0.2 meter. If the resulted cluster has more than 200 points (assuming a 4096 point sample in a 1m by 1m area), the cluster’s bounding box is marked as one object proposal. For each proposed object, it’s detection score is computed as the average point score for that category. Before evaluation, proposals with extremely small areas/volumes are pruned. For tables, chairs and sofas, the bounding boxes are extended to the floor in case the legs are separated with the seat/surface.`

많은 물체가 있어 가까운 경우에는 CC 알고리즘의 분할이 잘 동작 하지 않는다. 따라서, **sliding shape method** 방식을 이용하여 위자 분류 분제를 해결 하였다. `We observe that in some rooms such as auditoriums lots of objects (e.g. chairs) are close to each other, where connected component would fail to correctly segment out individual ones. Therefore we leverage our classification network and uses sliding shape method to alleviate the problem for the chair class. `

각 분류에 대해 바이너리 분류기를 학습 시켰다. We train a binary classification network for each category and use the classifier for sliding window detection. 결과 박스는 **non-maximum suppression**을 통해 제거 해 나갔다. CC와 **sliding shapes**를 통해 후보 박스 영역은 최종 평가를 위해 합쳐 졌다. `The resulted boxes are pruned by non-maximum suppression. The proposed boxes from connected component and sliding shapes are combined for final evaluation.`

In Fig 11, we show the precision-recall curves for object detection. We trained six models, where each one of them is trained on five areas and tested on the left area. At test phase, each model is tested on the area it has never seen. The test results for all six areas are aggregated for the PR curve generation.

---

# PointNet

- 스탠포드대 : http://stanford.edu/~rqi/pointnet/
- keras : https://github.com/garyli1019/pointnet-keras
- pytorch : https://github.com/fxia22/pointnet.pytorch
- Tensorflow : https://github.com/charlesq34/pointnet
- open3d활용 pytorch : https://github.com/IntelVCL/Open3D-PointNet
- Semantic3D (semantic-8) segmentation with Open3D and PointNet++ : https://github.com/IntelVCL/Open3D-PointNet2-Semantic3D
- pointnet-autoencoder : https://github.com/charlesq34/pointnet-autoencoder


This dataset provides part segmentation to a subset of ShapeNetCore models, containing ~16K models from 16 shape categories. The number of parts for each category varies from 2 to 6 and there are a total number of 50 parts.
The dataset is based on the following work:
```
@article{yi2016scalable,
  title={A scalable active framework for region annotation in 3d shape collections},
  author={Yi, Li and Kim, Vladimir G and Ceylan, Duygu and Shen, I and Yan, Mengyan and Su, Hao and Lu, ARCewu and Huang, Qixing and Sheffer, Alla and Guibas, Leonidas and others},
  journal={ACM Transactions on Graphics (TOG)},
  volume={35},
  number={6},
  pages={210},
  year={2016},
  publisher={ACM}
}
```
You could find the initial mesh files from the released version of ShapeNetCore.
An mapping from synsetoffset to category name could be found in "synsetoffset2category.txt"
```
The folder structure is as below:
	-synsetoffset
		-points : *.pts , x,y,z(??)
			-uniformly sampled points from ShapeNetCore models
		-point_labels : *.seg (1~3)
			-per-point segmentation labels
		-seg_img : *.png
			-a visualization of labeling
	-train_test_split
		-lists of training/test/validation shapes shuffled across all categories (from the official train/test split of ShapeNet)

    Airplane	02691156
    Bag	02773838
    Cap	02954340
    Car	02958343
    Chair	03001627
    Earphone	03261776
    Guitar	03467517
    Knife	03624134
    Lamp	03636649
    Laptop	03642806
    Motorbike	03790512
    Mug	03797390
    Pistol	03948459
    Rocket	04099429
    Skateboard	04225987
    Table	04379243
```


####   pts handling : [pts_loader](https://github.com/albanie/pts_loader)

```python
def load(path):
    """takes as input the path to a .pts and returns a list of 
	tuples of floats containing the points in in the form:
	[(x_0, y_0, z_0),
	 (x_1, y_1, z_1),
	 ...
	 (x_n, y_n, z_n)]"""
    with open(path) as f:
        rows = [rows.strip() for rows in f]
    
    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return points
```

#### pts Visualization : [A library for visualization and creative-coding ](https://github.com/williamngan/pts)

#### open3d지원

```python
pcd = read_point_cloud('./sample.pts')
```
