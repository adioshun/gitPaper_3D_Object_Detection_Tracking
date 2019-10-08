| 논문명 |PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space|
| --- | --- |
| 저자\(소속\) | Charles R. Qi \(Stanford University\)|
| 학회/년도 | NIPS'17,  [논문](https://arxiv.org/pdf/1706.02413.pdf) |
| 키워드 |   |
| 데이터셋(센서)/모델| ModelNet40,  ShapeNet part data, Stanford 3D semantic parsing data|
|관련연구|이후연구 : Pointnet++|
| 참고 | [ppt](https://www.facebook.com/thinking.factory/posts/1408857052524274), [홈페이지](http://stanford.edu/~rqi/pointnet/),[CVPR2017](https://www.youtube.com/watch?v=Cge-hot0Oc0), [Youtube](https://youtu.be/8CenT_4HWyY?t=1h16m39s), [ppt](http://3ddl.stanford.edu/CVPR17_Tutorial_PointCloud.pdf), [한글설명](http://daddynkidsmakers.blogspot.com/2017/07/3-pointnet.html)  |
| 코드 | [TF](https://github.com/charlesq34/pointnet), [pyTorch](https://github.com/fxia22/pointnet.pytorch)|







[Arxiv](https://arxiv.org/pdf/1706.02413.pdf), [홈페이지](http://stanford.edu/~rqi/pointnet2/)


PointNet++ : [TF-Official](https://github.com/charlesq34/pointnet2), [TF-Open3D](https://github.com/intel-isl/Open3D-PointNet2-Semantic3D), [TF-Open3D-설명](http://www.open3d.org/index.php/2019/01/16/on-point-clouds-semantic-segmentation/)

도커 :  [PointNet++](https://hub.docker.com/r/zjuncd/pointnet2)


---


기존 연구는 local structures를 고려 하지 않았다. 따라서 ine-grained patterns를 인식 하지 못하고 복잡한 환경에서 일반화 성능이 떨어 진다. 본 논문에서는 계측정 네트워크를 소개 한다.  metric space distances정보를 활용함으로써 환경의 scale적 요소를 고려한 Locarl Feature를 학습 할수 있다. 또한 포인트 클라우드는 밀집정도가 다양함으로 uniform한 밀집도를 가진 데이터 에서 학습시 성능이 떨어 진다. 이를 해결 하기 위한 기법도 적용 하였다. `Few prior works study deep learning on point sets. PointNet [20] is a pioneer in this direction. However, by design PointNet does not capture local structures induced by the metric space points live in, limiting its ability to recognize fine-grained patterns and generalizability to complex scenes. In this work, we introduce a hierarchical neural network that applies PointNet recursively on a nested partitioning of the input point set. By exploiting metric space distances, our network is able to learn local features with increasing contextual scales. With further observation that point sets are usually sampled with varying densities, which results in greatly decreased performance for networks trained on uniform densities, we propose novel set learning layers to adaptively combine features from multiple scales. Experiments show that our network called PointNet++ is able to learn deep point set features efficiently and robustly. In particular, results significantly better than state-of-the-art have been obtained on challenging benchmarks of 3D point clouds.`

## 1 Introduction

3D 환경 분석에서 포인트 클라우드는 중요 하다. 이러한 데이터는 **invariant to permutations of its members**하고  **dthe distance metric defines local neighborhoods that may exhibit different properties**한다.  예를 들어 서로 다른 위치에서 밀집도나 다른 특성들이 균일 하지 않을수 있다. ` We are interested in analyzing geometric point sets which are collections of points in a Euclidean space. A particularly important type of geometric point set is point cloud captured by 3D scanners, e.g., from appropriately equipped autonomous vehicles. As a set, such data has to be invariant to permutations of its members. In addition, the distance metric defines local neighborhoods that may exhibit different properties. For example, the density and other attributes of points may not be uniform across different locations — in 3D scanning the density variability can come from perspective effects, radial density variations, motion, etc.`


이전 PointNet은 각 포인트의 특징을 모두 합쳐서 전역 특징을 만든다. 이 방식은 Metric기반의 local structure를 파악 하지 못한다. 하지만 local structure를 이용하는것은 컨볼류션 구조에서 중요하다. `Few prior works study deep learning on point sets. PointNet [20] is a pioneering effort that directly processes point sets. The basic idea of PointNet is to learn a spatial encoding of each point and then aggregate all individual point features to a global point cloud signature. By its design, PointNet does not capture local structure induced by the metric. However, exploiting local structure has proven to be important for the success of convolutional architectures.`

CNN은 그리드로 정의된 입력 데이터에서 Low/High level에 따른 receptive fields를 사용한다. 이 방식을 통해 처음 접하는 환경에서도 좋은 성능을 보인다. ` A CNN takes data defined on regular grids as the input and is able to progressively capture features at increasingly larger scales along a multi-resolution hierarchy. At lower levels neurons have smaller receptive fields whereas at higher levels they have larger receptive fields. The ability to abstract local patterns along the hierarchy allows better generalizability to unseen cases.`

본 논문에서는 계측적 성격으로 가지며 metric space에서 포인트 샘플링을 수행 한는 PointNET2를 제안 한다. `We introduce a hierarchical neural network, named as PointNet++, to process a set of points sampled in a metric space in a hierarchical fashion.`

기본 아이디어는 아래와 같다. ` The general idea of PointNet++ is simple.`
- We first partition the set of points into overlapping local regions by the distance metric of the underlying space. 
- Similar to CNNs, we extract local features capturing fine geometric structures from small neighborhoods; such local features are further grouped into larger units and processed to produce higher level features. 
- This process is repeated until we obtain the features of the whole point set.

제안 방식은 두가지 물음에 대한 답을 제공 할것이다. `The design of PointNet++ has to address two issues: `
- 파티션을 어떻게 나눌것인가? `how to generate the partitioning of the point set, and `
- 지역 특징을 어떻게 학습할 것인가? `how to abstract sets of points or local features through a local feature learner. `

위 두 문제는 서로 연관이 있다. 지역 특징 학습을 위해 이전 PointNet을 사용할것이다. 기본 블록으로써 PointNet은 지역포인트/특징을 상위 레벨의 representations으로 변경 시킨다. 이런관점에서 PointNet++는 PointNet의 재귀버젼이다. ` The two issues are correlated because the partitioning of the point set has to produce common structures across partitions, so that weights of local feature learners can be shared, as in the convolutional setting. We choose our local feature learner to be PointNet. As demonstrated in that work, PointNet is an effective architecture to process an unordered set of points for semantic feature extraction. In addition, this architecture is robust to input data corruption. As a basic building block, PointNet abstracts sets of local points or features into higher level representations. In this view, PointNet++ applies PointNet recursively on a nested partitioning of the input set.`

또다른 이슈는 **어떻게 겹쳐진 파티션들을 생성 하는가?** 이다. 각 파티션은 **neighborhood ball(중심점, 크기)**로 정의 한다. 모든 공간을 고르게 커버 하기 위해서 **중심점**은  farthest point sampling (FPS)알고리즘을 이용하여 구한다. 기존 CNN에서 고정된 strides값으로 스캔하는것 대비 제안 방식의 **local receptive fields **는 입력 데이터/Metric  정보를 이용하므로 더 효율적이다. ` One issue that still remains is how to generate overlapping partitioning of a point set. Each partition is defined as a neighborhood ball in the underlying Euclidean space, whose parameters include centroid location and scale. To evenly cover the whole set, the centroids are selected among input point set by a farthest point sampling (FPS) algorithm. Compared with volumetric CNNs that scan the space with fixed strides, our local receptive fields are dependent on both the input data and the metric, and thus more efficient and effective.`

**neighborhood ball(중심점, 크기)**의 **크기**를 구하는건 **entanglement of feature scale**와 **non-uniformity of input point set** 때문에 좀더 어렵다. 일반적으로 포인트 클라우드는 원거리에 있을경우 조밀도가 낮다. 이는 이미지 기반 CNN에서는 고려 하지 않아도 되는 상황이다. `Deciding the appropriate scale of local neighborhood balls, however, is a more challenging yet intriguing problem, due to the entanglement of feature scale and non-uniformity of input point set. We assume that the input point set may have variable density at different areas, which is quite common in real data such as Structure Sensor scanning [18] (see Fig. 1). Our input point set is thus very different from CNN inputs which can be viewed as data defined on regular grids with uniform constant density. In CNNs, the counterpart to local partition scale is the size of kernels. [25] shows that using smaller kernels helps to improve the ability of CNNs. Our experiments on point set data, however, give counter evidence to this rule. Small neighborhood may consist of too few points due to sampling deficiency, which might be insufficient to allow PointNets to capture patterns robustly.`

본 논문의 기여도는 아래와 같다. `A significant contribution of our paper is that `
- PointNet++ leverages neighborhoods at multiple scales to achieve both robustness and detail capture. 
- Assisted with random input dropout during training, the network learns to adaptively weight patterns detected at different scales and combine multi-scale features according to the input data. 

성능평가 결과도 좋음 `Experiments show that our PointNet++ is able to process point sets efficiently and robustly. In particular, results that are significantly better than state-of-the-art have been obtained on challenging benchmarks of 3D point clouds.`


## 2 Problem Statement

Suppose that X = (M, d) is a discrete metric space whose metric is inherited from a Euclidean space `R^n`, 
- where M ⊆ R^n is the set of points and 
- d is the distance metric. 

In addition, the density of M in the ambient Euclidean space may not be uniform everywhere. We are interested in learning set functions f that take such X as the input (along with additional features for each point) and produce information of semantic interest regrading X . In practice, such f can be classification function that assigns a label to X or a segmentation function that assigns a per point label to each member of M.

## 3 Method

본 논문의 방식은 기존 PointNet을 계층적 구조로 확장한 개념이다. `Our work can be viewed as an extension of PointNet [20] with added hierarchical structure. `

- We first review PointNet (Sec. 3.1) and 
- then introduce a basic extension of PointNet with hierarchical structure (Sec. 3.2). 
- Finally, we propose our PointNet++ that is able to robustly learn features even in non-uniformly sampled point sets (Sec. 3.3).

### 3.1 Review of PointNet [20]: A Universal Continuous Set Function Approximator

입력 포인트를 벡터로 맵핑하는 함수 **f()**는 다음과 같다. `Given an unordered point set {x1, x2, ..., xn} with x_i ∈ R^d , one can define a set function f : X → R that maps a set of points to a vector:`

![](https://i.imgur.com/8QXTsy8.png)

- where `γ` and `h` are usually **multi-layer perceptron (MLP) networks**.

함수 `f()`는 포인트클라우드의 입력순서에 영향을 받지 않고 독립적(arbitrarily approximate)으로 동작 한다. `The set function f in Eq. 1 is invariant to input point permutations and can arbitrarily approximate any continuous set function [20]. `

함수 `h()`는 **spatial encoding ** 역할을 수행하는것으로 볼수 있다. `Note that the response of h can be interpreted as the spatial encoding of a point (see [20] for details).`

이전 포인트넷은 **different scales** 환경에서 성능이 좋지 않다. 이 제약 해결을 위해 다음장에서 계측정 특징 학습기를 설명 하겠다. `PointNet achieved impressive performance on a few benchmarks. However, it lacks the ability to capture local context at different scales. We will introduce a hierarchical feature learning framework in the next section to resolve the limitation.`

### 3.2 Hierarchical Point Set Feature Learning

기존방식에서는 단일 Max Pooling 방식으로 모든 포인트의 특징을 모았지만, 개선 방식에서는 포인트의 계층적 그룹을 만들고 progressively abstract larger and larger local regions along the hierarchy 한다. `While PointNet uses a single max pooling operation to aggregate the whole point set, our new architecture builds a hierarchical grouping of points and progressively abstract larger and larger local regions along the hierarchy.`

![](https://i.imgur.com/24gbHQl.png)


계층적 구성은 여러개의 **abstraction levels**로 구성되어 있다. `Our hierarchical structure is composed by a number of set abstraction levels (Fig. 2). At each level, a set of points is processed and abstracted to produce a new set with fewer elements. `

구성 요소는 세가지 이다. `The set abstraction level is made of three key layers: Sampling layer, Grouping layer and PointNet layer. `
- The Sampling layer selects a set of points from input points, which defines the centroids of local regions. 
- Grouping layer then constructs local region sets by finding “neighboring” points around the centroids. 
- PointNet layer uses a mini-PointNet to encode local region patterns into feature vectors.

A set abstraction level 
- 입력 : takes an N × (d + C) matrix as input that is from N points with d-dim coordinates and C-dim point feature. 
- 출력 : It outputs an N0 × (d + C 0 ) matrix of N0 subsampled points with d-dim coordinates and new C 0 -dim feature vectors summarizing local context. 

We introduce the layers of a set abstraction level in the following paragraphs.

#### A. Sampling layer

Given input points {x1, x2, ..., xn}, we use iterative farthest point sampling (FPS) to choose a subset of points {xi1 , xi2 , ..., xim}, such that xij is the most distant point (in metric distance) from the set {xi1 , xi2 , ..., xij−1 } with regard to the rest points. 

랜덤 방식과 비교 하여 제안 방식이 모든 포인트 영역을 커버 한다. CNN방식(scan the vector space agnostic of data distribution,)과 반대로 제안 샘플링 방식은 데이터 의존적적으로 **receptive fields**를 생성 한다. `Compared with random sampling, it has better coverage of the entire point set given the same number of centroids. In contrast to CNNs that scan the vector space agnostic of data distribution, our sampling strategy generates receptive fields in a data dependent manner.`

#### B. Grouping layer

입력 : The input to this layer is a point set of size N × (d + C) and the coordinates of a set of centroids of size N' × d. 

출력 : The output are groups of point sets of size N' × K × (d + C), 
- where each group corresponds to a local region and K is the number of points in the neighborhood of centroid points. 
- Note that K varies across groups but the succeeding PointNet layer is able to convert flexible number of points into a fixed length local region feature vector.

기존 CNN은 픽셀 기반이지만, 제안 방식은 미터 거리 기반 이다. In convolutional neural networks, a local region of a pixel consists of pixels with array indices within certain Manhattan distance (kernel size) of the pixel. In a point set sampled from a metric space, the neighborhood of a point is defined by metric distance.

**Ball query** 방식은 주어진 포인트에서 반경내 모든 포인트를 찾아 낸다. 비슷한 방식은 KNN이 있는데 이 방식은 주어진 숫자 만큰의 이웃 포인트 들을 찾아 낸다. KNN에 비해 주어진 방식이 로컬 패턴인식에 더 효율적이다. `Ball query finds all points that are within a radius to the query point (an upper limit of K is set in implementation). An alternative range query is K nearest neighbor (kNN) search which finds a fixed number of neighboring points. Compared with kNN, ball query’s local neighborhood guarantees a fixed region scale thus making local region feature more generalizable across space, which is preferred for tasks requiring local pattern recognition (e.g. semantic point labeling).`

#### C. PointNet layer

입력 : In this layer, the input are N' local regions of points with data size N'×K×(d+C). 
출력 : Each local region in the output is abstracted by its centroid and local feature that encodes the centroid’s neighborhood. Output data size is N' × (d + C' ).

The coordinates of points in a local region are firstly translated into a local frame relative to the centroid point: x (j) i = x (j) i − xˆ (j) for i = 1, 2, ..., K and j = 1, 2, ..., d where xˆ is the coordinate of the centroid. 

We use PointNet [20] as described in Sec. 3.1 as the basic building block for local pattern learning. By using relative coordinates together with point features we can capture point-to-point relations in the local region.


### 3.3 Robust Feature Learning under Non-Uniform Sampling Density

밀집도가 다른것에서 학습한것은 다른 밀집도 환경에 적용하기 어렵다. `As discussed earlier, it is common that a point set comes with nonuniform density in different areas. Such non-uniformity introduces a significant challenge for point set feature learning. Features learned in dense data may not generalize to sparsely sampled regions. Consequently, models trained for sparse point cloud may not recognize fine-grained local structures.`




이 문제 해결을 위해 **density adaptive PointNet layers**을 제안 하였다. 이 방식은 입력 샘플의 밀집도가 다른것에 대응 하도록 되어 있다. `Ideally, we want to inspect as closely as possible into a point set to capture finest details in densely sampled regions. However, such close inspect is prohibited at low density areas because local patterns may be corrupted by the sampling deficiency. In this case, we should look for larger scale patterns in greater vicinity. To achieve this goal we propose density adaptive PointNet layers (Fig. 3) that learn to combine features from regions of different scales when the input sampling density changes. We call our hierarchical network with density adaptive PointNet layers as PointNet++.`

Previously in Sec. 3.2, each abstraction level contains grouping and feature extraction of a single scale. In PointNet++, each abstraction level extracts multiple scales of local patterns and combine them intelligently according to local point densities. 

**local regions 그룹핑**과 **다른 크기의 특징을 결합** 하기 위해 두 종류의 **density adaptive layers **를 제안 하였다.  `In terms of grouping local regions and combining features from different scales, we propose two types of density adaptive layers as listed below.`

![](https://i.imgur.com/G3VnFYi.png)

#### Density Adaptive Layers #1 : Multi-scale grouping (MSG)

> 간단 하고 성능 좋지만,  계산 부하가 크다. 

가장 간단하면서도 효율적인 방법은 PointNet으로 각 scale별로 특징을 추출한후 서로 다른 scale별로 그룹핑 레이어를 적용 하는것이다. 서로 다른 scale의 특징들은 **multi-scale feature** 형태로 합쳐지게 된다. `As shown in Fig. 3 (a), a simple but effective way to capture multiscale patterns is to apply grouping layers with different scales followed by according PointNets to extract features of each scale. Features at different scales are concatenated to form a multi-scale feature.`

**random input dropout** 기법 적용 : We train the network to learn an optimized strategy to combine the multi-scale features. This is done by randomly dropping out input points with a randomized probability for each instance, which we call random input dropout. Specifically, for each training point set, we choose a dropout ratio θ uniformly sampled from [0, p] where p ≤ 1. For each point, we randomly drop a point with probability θ. In practice we set p = 0.95 to avoid generating empty point sets. In doing so we present the network with training sets of various sparsity (induced by θ) and varying uniformity (induced by randomness in dropout). During test, we keep all available points.



#### Density Adaptive Layers #2 : Multi-resolution grouping (MRG)

위 MSG방식은 계산 부하가 크다. `The MSG approach above is computationally expensive since it runs local PointNet at large scale neighborhoods for every centroid point. In particular, since the number of centroid points is usually quite large at the lowest level, the time cost is significant.`

성능 차이는 없지만 부하가 적은 방식 제안 `Here we propose an alternative approach that avoids such expensive computation but still preserves the ability to adaptively aggregate information according to the distributional properties of points. `

특정 레벨에서의 영역 특징은 두가지 벡터를 합친다. `In Fig. 3 (b), features of a region at some level L_i is a concatenation of two vectors. `
- One vector (left in figure) is obtained by summarizing the features at each subregion from the lower level Li−1 using the set abstraction level. 
- The other vector (right) is the feature that is obtained by directly processing all raw points in the local region using a single PointNet.

밀집도가 낮으면 두번째 벡터가 중요하고, 반대면 첫번째 백터가 중요하므로 가중치를 준다. `When the density of a local region is low, the first vector may be less reliable than the second vector, since the subregion in computing the first vector contains even sparser points and suffers more from sampling deficiency. In such a case, the second vector should be weighted higher. On the other hand, when the density of a local region is high, the first vector provides information of finer details since it possesses the ability to inspect at higher resolutions recursively in lower levels.`

MSG방식대비 처리 부하가 적다 `Compared with MSG, this method is computationally more efficient since we avoids the feature extraction in large scale neighborhoods at lowest levels.`

### 3.4 Point Feature Propagation for Set Segmentation

**abstraction layer**에서 원본 포인트는 서브-샘플링된다. 하지만 세그멘테이션을 위해서는 각 포인트별 특징정보가 필요 하다. `In set abstraction layer, the original point set is subsampled. However in set segmentation task such as semantic point labeling, we want to obtain point features for all the original points.`
- 해결책 #1 :  모든 포인트를 Centroid로 하여 샘플링 하는것이다. 계산 부하가 크다. `One solution is to always sample all points as centroids in all set abstraction levels, which however results in high computation cost.`
- 해결책 #2 : 서브-샘플에서 원본 포인트로 특징을 전파 하는 것이다. ` Another way is to propagate features from subsampled points to the original points.`

![](https://i.imgur.com/24gbHQl.png)

본 논문에서는 계층적 전파 방식(with 거리기반 보간법 + across level skip links)을 사용하였다. `We adopt a hierarchical propagation strategy with distance based interpolation and across level skip links (as shown in Fig. 2).`

-  In a feature propagation level, we propagate point features from Nl × (d + C) points to Nl−1 points where Nl−1 and Nl (with Nl ≤ Nl−1) are point set size of input and output of set abstraction level l. 
- We achieve feature propagation by interpolating feature values f of Nl points at coordinates of the Nl−1 points. 
- Among the many choices for interpolation, we use inverse distance weighted average based on k nearest neighbors (as in Eq. 2, in default we use p = 2, k = 3). 
- The interpolated features on Nl−1 points are then concatenated with skip linked point features from the set abstraction level. 
- Then the concatenated features are passed through a “unit pointnet”, which is similar to one-by-one convolution in CNNs.
-  A few shared fully connected and ReLU layers are applied to update each point’s feature vector. 
- The process is repeated until we have propagated features to the original set of points.

![](https://i.imgur.com/p4uuXsl.png)


## 4 Experiments


## 5 Related Work

계층적으로 특징을 학습 하는 방식은 좋은 성과를 보였다. 학습 모델 중에서 CNN방식은 가장 성능이 좋다. 하지만 CNN은 **거리정**를 가지는 **unordered point sets**를 처리에는 적합하지 않다. 이 점이 본 논문의 시작이다. ` The idea of hierarchical feature learning has been very successful. Among all the learning models, convolutional neural network [10, 25, 8] is one of the most prominent ones. However, convolution does not apply to unordered point sets with distance metrics, which is the focus of our work.`


unordered sets에 딥러닝을 적용하는 방법에 대한 연구가 있지만[20,28] **distance metric**를 고려 하지 않았다. 결과적으로 이 방식들은 제약을 가지고 있다. `A few very recent works [20, 28] have studied how to apply deep learning to unordered sets. They ignore the underlying distance metric even if the point set does possess one. As a result, they are unable to capture local context of points and are sensitive to global set translation and normalization. In this work, we target at points sampled from a metric space and tackle these issues by explicitly considering the underlying distance metric in our design.`

거리와 밀집도가 다른 문제에 대한 연구도 진행 되어 왔다[19, 17, 2, 6, 7, 30]. `Point sampled from a metric space are usually noisy and with non-uniform sampling density. This affects effective point feature extraction and causes difficulty for learning. One of the key issue is to select proper scale for point feature design. Previously several approaches have been developed regarding this [19, 17, 2, 6, 7, 30] either in geometry processing community or photogrammetry and remote sensing community. In contrast to all these works, our approach learns to extract point features and balance multiple feature scales in an end-to-end fashion.`

3D 공간 표현(representations)에 **volumetric grids**나 **geometric graphs ** 같은 방식도 있지만 이 방식들은 거리와 밀집도 문제를 다루지는 않았다. `In 3D metric space, other than point set, there are several popular representations for deep learning, including volumetric grids [21, 22, 29], and geometric graphs [3, 15, 33]. However, in none of these works, the problem of non-uniform sampling density has been explicitly considered.`

## 6 Conclusion

In this work, we propose PointNet++, a powerful neural network architecture for processing point sets sampled in a metric space. PointNet++ recursively functions on a nested partitioning of the input point set, and is effective in learning hierarchical features with respect to the distance metric. To handle the non uniform point sampling issue, we propose two novel set abstraction layers that intelligently aggregate multi-scale information according to local point densities. These contributions enable us to achieve state-of-the-art performance on challenging benchmarks of 3D point clouds. 

In the future, it’s worthwhile thinking how to accelerate inference speed of our proposed network especially for MSG and MRG layers by sharing more computation in each local regions. It’s also interesting to find applications in higher dimensional metric spaces where CNN based method would be computationally unfeasible while our method can scale well.

---

# Supplementary

## A Overview

This supplementary material provides more details on experiments in the main paper and includes more experiments to validate and analyze our proposed method. 
- In Sec B we provide specific network architectures used for experiments in the main paper and also describe details in data preparation and training. 
- In Sec C we show more experimental results including benchmark performance on part segmentation and analysis on neighborhood query, sensitivity to sampling randomness and time space complexity.

## B Details in Experiments

```
- SA is a set abstraction (SA) level with K local regions of ball radius r using PointNet of d fully connected layers with width li (i = 1, ..., d). 
	- SA is a global set abstraction level that converts set to a single vector. 
- In multi-scale setting (as in MSG), 
	- we use SA to represent MSG with m scales. 
- FC represents a fully connected layer with width l and dropout ratio dp. 
- FP is a feature propagation (FP) level with d fully connected layers. 
	- It is used for updating features concatenated from interpolation and skip link. 
- All fully connected layers are followed by batch normalization and ReLU except for the last score prediction layer.


### B.1 Network Architectures



```



### 


---

-  [거리공간이란? Metric Space](https://freshrimpsushi.tistory.com/381)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTcxNDExMTMxMl19
-->