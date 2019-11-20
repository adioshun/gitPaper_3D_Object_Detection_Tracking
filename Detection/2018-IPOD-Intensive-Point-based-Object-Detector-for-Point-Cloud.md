# [IPOD: Intensive Point-based Object Detector for Point Cloud](https://arxiv.org/abs/1812.05276)

We present a novel 3D object detection framework, named IPOD, based on raw point cloud. It seeds object proposal for each point, which is the basic element. This paradigm provides us with high recall and high fidelity of information, leading to a suitable way to process point cloud data. We design an end-to-end trainable architecture, where features of all points within a proposal are extracted from the backbone network and achieve a proposal feature for final bounding inference. These features with both context information and precise point cloud coordinates yield improved performance. We conduct experiments on KITTI dataset, evaluating our performance in terms of 3D object detection, Bird’s Eye View (BEV) detection and 2D object detection. Our method accomplishes new state-of-the-art , showing great advantage on the hard set.


## 1. Introduction

본 논문에서는 물체 탐지, 분류에 대하여 다루겠다. `Great breakthrough has been made in 2D image recognition tasks [14, 25, 24, 9] with the development of Convolutional Neural Networks (CNNs). Meanwhile, 3D scene understanding with point cloud also becomes an important topic, since it can benefit many applications, such as autonomous driving [13] and augmented reality [27]. In this work, we focus on one of the most important 3D scene recognition tasks – that is, 3D object detection based on point cloud, which predicts the 3D bounding box and class label for each object in the scene.`

### 1.1 Challenges 

라이다 데이터는 3가지 특징을 가지고 있으며 투영이나 복셀화를 통한 기존 2D CNN을 적용해도 성능이 좋지 않다. `Different from RGB images, LiDAR point cloud is with its unique properties. On the one hand, they provide more spatial and structural information including precise depth and relative location. On the other hand, they are sparse, unordered, and even not uniformly distributed, bringing huge challenge to 3D recognition tasks. To deploy CNNs, most existing methods convert 3D point clouds to images by projection [4, 19, 15, 28, 10] or voxelize cloud with a fixed grid [26, 32, 35]. With the compact representation, CNN is applied. Nevertheless, these hand-crafted representations may not be always optimal regarding the detection performance.`

다른 도전으로는 2D 탐지기로 후보 영역을 선발하고 PointNet을 적용하는 것이다. `Along another line, FPointNet [29] crops point cloud in a frustum determined by 2D object detection results. Then a PointNet [30] is applied on each frustum to produce 3D results. The performance of this pipeline heavily relies on the image detection results. Moreover, it is easily influenced by large occlusion and clutter objects, which is the general weakness of 2D object detectors.`

### 1.2 Our Contribution 

본 논문에서는 각 포인트를 **요소**로 다루고 후보영역 추출의 SEED로 사용한다. `To address aforementioned drawbacks, we propose a new paradigm based on raw point cloud for 3D object detection. We take each point in the cloud as the element and seed them with object proposals. `

The raw point cloud, without any approximation, is taken as input to keep sufficient information. This design is general and fundamental for point cloud data, and is able to handle occlusion and clutter scenes. We note it is nontrivial to come up with such a solution due to the well-known challenges of heavily redundant proposals and ambiguity on assigning corresponding groundtruth labels. 

Our novelty is on a proposal generation module to output proposals based on each point and effective selection of representative object proposals with corresponding ground-truth labels to ease network training. 

Accordingly, the new structure extracts both context and local information for each proposal, which is then fed to a tiny PointNet to infer final results. 

성능평가 결과가 좋다. 특히 **difficult examples** 분류에 좋은 성과를 보인다. `We evaluate our model on 2D detection, Bird’s Eye View (BEV) detection, and 3D detection tasks on KITTI benchmark [1]. Experiments show that our model outperforms state-of-the-art LIDAR based 3D object detection frameworks especially for difficult examples. `

특히 재현율이 성능이 좋다. `Our experiments also surprisingly achieve extremely high recall without the common projection operations. Our primary contribution is manifold.`

- 후보 생성 방법 제안 `We propose a new proposal generation paradigm for point cloud based object detector. It is a natural and general design, which does not need image detection while yielding much higher recall compared with widely used voxel and projection-based methods.`

- A network structure with input of raw point cloud is proposed to produce features with both context and local information.

- Experiments on KITTI datasets show that our framework better handles many hard cases with highly occluded and crowded objects, and achieves new stateof-the-art performance.

## 2. Related Work

### 2.1 3D Semantic Segmentation

There have been several approaches to tackle semantic segmentation on point cloud. 

- In [33], a projection function converts LIDAR points to a UV map, which is then classified by 2D semantic segmentation [33, 34, 3] in pixel level. 

- In [8, 7], a multi-view based function produces the segmentation mask. The method fuses information from different views. 


- Other solutions, such as [31, 30, 22, 17, 21], segment the point cloud from raw LIDAR data. 
	- They directly generate features on each point while keeping original structural information. 
	- Specifically, a max-pooling method gathers the global feature; 
	- it is then concatenated with local feature for processing.

### 2.2 3D Object Detection 

There are roughly three different lines for 3D object detection. They are voxel-grid based, multi-view based and PointNet based methods.

#### A. Voxel-grid Method: 

There are several LIDAR-data based 3D object detection frameworks using voxel-grid representation. In [32], each non-empty voxel is encoded with 6 statistical quantities by the points within this voxel. A binary encoding is used in [20] for each voxel grid. They utilized hand-crafted representation. VoxelNet [35] instead stacks many VFE layers to generate machine-learned representation for each voxel.


#### B. Multi-view Method: 

MV3D [4] projected LIDAR point cloud to BEV and trained a Region Proposal Network (RPN) to generate positive proposals. Afterwards, it merged features from BEV, image view and front view in order to generate refined 3D bounding boxes. AVOD [19] improved MV3D by fusing image and BEV features like [23]. Unlike MV3D, which only merges features in the refinement phase, it also merges features from multiple views in the RPN phase to generate more accurate positive proposals. However, these methods still have the limitation when detecting small objects such as pedestrians and cyclists. They do not handle several cases that have multiple objects in depth direction.


#### C. P ointNet Method:

 F-PointNet [29] is the first method of utilizing raw point cloud to predict 3D objects. Initially, a 2D object detection module [11] is applied to generate frustum proposals. Then it crops points and passes them into an instance segmentation module. At last, it regresses 3D bounding boxes by the positive points output from the segmentation module. Final performance heavily relies on the detection results from the 2D object detector. In contrast, our design is general and effective to utilize the strong representation power of point cloud.

## 3. Our Framework


![](https://i.imgur.com/i0gCv6z.png)
```
Figure 1. Illustration of our framework. It consists of three different parts. 
- The first is a subsampling network to filter out most background points. 
- The second part is for point-based proposal generation. 
- The third component is the network architecture, which is composed of backbone network, proposal feature generation module and a box prediction network. It classifies and regresses generated proposals.
```

제안 방식의 목적은 후보 영역을 생성하고 BBox를 유추(regress) 하는 것이다. `Our method aims to regress the 3D object bounding box from the easy-to-get point-based object proposals, which is a natural design for point cloud based object detection. `

To make it feasible, we design new strategies to reduce redundancy and ambiguity existing introduced by trivially seeding proposals for each point. After generating proposals, we extract features for final inference. Our framework is illustrated in Figure 1.

### 3.1. Point-based Proposal Generation

점군으 unordered한 속성을 가진다. `The point cloud gathered from LiDAR is unordered, making it nontrivial to utilize the powerful CNN to achieve good performance. `

기존 방식은 투영하거나 복셀화 하여 구조를 생성(unordered->ordered)하는 것이다. `Existing methods mainly project point cloud to different views or divide them into voxels, transforming them into a compact structure. `

We instead choose a more general strategy to seed object proposals based on each point independently, which is the elementary component in the point cloud. Then we process raw point cloud directly. 

As a result, precise localization information and a high recall are maintained, i.e., achieving a 96.0% recall on KITTI[12] dataset.

#### A. Challenges 

Point기반 처리가 Elegant하긴 하지만 많은 문제점이 있다. `Albeit elegant, point-based frameworks inevitably face many challenges. `
- For example, the amount of points is prohibitively huge and high redundancy exists between different proposals. 
- They cost much computation during training and inference. 
- Also, ambiguity of regression results and assigning ground-truth labels for nearby proposals need to be resolved.

#### B. Selecting Positive Points 
![](https://i.imgur.com/Q7SyHuV.png)
```
Figure 2. Illustration of point-based proposal generation. 
- (a) Semantic segmentation result on the image. 
- (b) Projected segmentation result on point cloud. 
- (c) Point-based proposals on positive points after NMS.
```


첫번째는 **배경 점군을 제거** 하는 것이다. `The first step is to filter out background points in our framework. `

We use a 2D semantic segmentation network named subsampling network to predict the foreground pixels and then project them into point cloud as a mask with the given camera matrix to gather positive points. 

As shown in Figure 2, the positive points within a bounding box are clustered. 

We generate proposals at the center of each point with multiple scales, angles and shift, which is illustrated in Figure 3. 

These proposals can cover most of the points within a car.

![](https://i.imgur.com/ybmq4KM.png)
```
Figure 3. Illustration of proposal generation on each point from BEV. 
- We totally generate 6 proposals based on 2 different anchors with angle of 0 or 90 degree. 
- For each fundamental anchor, we use 3 different shifts along horizontal axis at ratios of -0.5, 0 and 0.5.
```

#### C. Reducing Proposal Redundancy 

배경 제거후 남은 60,000개의 포인중 대부분이 불필요한것 것들이다. NMS를 통해서 제거 하였다. `After background point removal, around 60K proposals are left; but many of them are redundant. We conduct non-maximum suppression (NMS) to remove the redundancy. `

The score for each proposal is the **sum of semantic segmentation scores** of interior points, making it possible to select proposals with a relatively large number of points. 

The intersection-over-union (IoU) value is calculated based on the **projection of each proposal to the BEV**. With these operations, we reduce the number of effective proposals to around 500 while keeping a high recall.

#### D. Reduction of Ambiguity 

![](https://i.imgur.com/tYcyljN.png)
```
Figure 4. Illustration of paradox situations. 
- (a) Different proposals with the same output. 
- (b) True positive proposal assigned to a negative label.
```

There are cases that two different proposals contain the same set of points, as shown in Figure 4(a). 

Since the feature for each proposal is produced using the interior points, these proposals thus possess the same feature representation, leading to the same classification or regression prediction and yet different bounding box regression results. 

To eliminate this contradiction, we align these two proposals by replacing their sizes and centers with predefined class-specific anchor size and the center of the set of interior points. 

As illustrated in Figure 4(a), the two different proposals A and B are aligned with these steps to proposal C. 

Another ambiguity lies on assigning target labels to proposals during training. It is not appropriate to assign positive and negative labels considering only IoU values between proposals and ground-truth boxes, as what is performed in 2D object detector. 

As shown by Figure 4(b), proposal A contains all points within a ground-truth box and overlaps with this box heavily. It is surely a good positive proposal. 

Contrarily, proposal B is with a low IoU value and yet contains most of the ground-truth points. With the criterion in 2D detector only considering box IoU, proposal B is negative. 

Note in our point-based settings, interior points are with much more importance. It is unreasonable to consider the bounding box IoU. 

Our solution is to design a new criterion named PointsIoU to assign target labels. PointsIoU is defined as the quotient between the number of points in the intersection area of both boxes and the number of points in the union area of both boxes. 

According to PointsIoU, both proposals in Figure 4(b) are now positive.

### 3.2. Network Architecture 

Accurate object detection requires the network to be able to produce correct class label with precise localization information for each instance. As a result, our network needs to be aware of context information to help classification and utilize fine-grained location in raw point cloud. 

Our network takes entire point cloud as input and produces the feature representation for each proposal. As shown in Figure 1, our network consists of a backbone, a proposal feature generation module and a box prediction network.

#### A. Backbone Network 

The backbone network based on **PointNet++ [31]** takes entire point cloud, with each point parameterized by coordinate and reflectance value, i.e., ([x, y, z, r]). 

The network is composed of a number of set abstraction (SA) levels and feature propagation (FP) layers, effectively gathering local features from neighboring points and enlarging the receptive field for each point. 

For N × 4 input points, the network outputs the feature map with size N × C where each row represents one point. 

Computation is shared by all different proposals, greatly reducing computation. 

Since features are generated from the raw points, no voxel projection is needed. The detail of our backbone network is shown in Figure 6(a).


#### B.  Proposal Feature Generation 
![](https://i.imgur.com/QTdtau9.png)
```
Figure 5. Illustration of proposal feature generation module. 
- It combines location information and context feature to generate offsets from the centroid of interior points to the center of target instance object. 
- The predicted residuals are added back to the location information in order to make feature more robust to geometric transformation.
```

The feature of each proposal has two parts, as shown in Figure 5. 

The first is cropped from the extracted feature map. 
- Specifically, for each proposal, we randomly select M = 512 points. 
- Then we take corresponding feature vector with size M × C and denoted it as F1. 
- With the SA and FP operations in PointNet++, these features well capture context information. 
- Besides this high-level abstraction, point location is also with great importance. 

Our second part is the proposal feature F2, the canonized coordinates of the M selected points. 
- Compared with the original coordinates, the canonized ones are more robust to the geometric transformation. 
- We utilize T-Net, which is one type of supervised Spatial Transformation Network (STN) [16], to calculate residuals from proposal center to real object center, denoted as ∆Cctr. 
- The input to T-Net is the concatenation of part 1 of proposal feature F1 and the M points’ corresponding XY Z coordinates, normalized by subtracting the center coordinates of these M points. 

As shown in Figure 5, part 2 of proposal feature is the canonized coordinates of these M points, calculated by normalized coordinates subtracting the center shift predicted by T-Net. 

The final proposal feature is the concatenation of F1 and F2.

#### C.  Bounding-Box Prediction Network 
![](https://i.imgur.com/mMM5l6o.png)
```
Figure 6. Illustration of two network architectures. 
- (a) Backbone architecture. It takes a raw point cloud (x, y, z, r) as input, and extracts both local and global features for each point by stacking SA layers and FP modules. 
- (b) Bounding-box prediction network. It takes the feature from proposal feature generation module as input and produces classification and regression prediction.
```


In this module, for each proposal, we use a small PointNet++ to predict its class, size ratio, center residual as well as orientation. 

Detailed structure of our prediction network is illustrated in Figure 6(b). 
- We utilize 3 SA modules with MLP layers for feature extraction. 
- Then average pooling is used to produce the global feature. 
- Two branches for regression and classification is applied. 
- For size ratio, we directly regress the ratio of ground-truth size and the proposal’s, parametrized by (tl , th, tw). 
- We further predict the shift (tx, ty, tz) from refined center by T-Net to ground-truth center. 

As a result, final center prediction is calculated by Cpro+∆Cctr+∆C ∗ ctr, where Cpro, ∆Cctr and ∆C ∗ ctr denote the center of proposal, prediction of T-Net and shift from bounding-box prediction network, respectively. 

For heading angle, we use a hybrid of classification and regression formulation following [29]. 

Specifically, we pre-define Na equally split angle bins and classify the proposal’s angle into different bins. 

Residual is regressed with respect to the bin value. Na is set to 12 in our experiments.


#### D.  Loss Functions 

We use a **multi-task loss** to train our network. 

The loss function is defined as Eq. (1), 
- where L_{cls} is the labeled classification loss. 
- L_{loc} denotes location regression loss, 
- L_{ang} and L_{cor} are angle and corner losses respectively. 
- si and ui are the predicted semantic score and ground-truth label for proposal i, respectively. 
- N_{cls} and N_{pos} are the number of proposals and positive samples.


> 후략 


## 4. Experiments

## 5. Concluding 

Remarks We have proposed a new method operating on raw points. We seed each point with proposals, without loss of precious localization information from point cloud data. Then prediction for each proposal is made on the proposal feature with context information captured by large receptive field and point coordinates that keep accurate shape information. Our experiments have shown that our model outperforms state-of-the-art 3D detection methods in hard set by a large margin, especially for those high occlusion or crowded scenes.