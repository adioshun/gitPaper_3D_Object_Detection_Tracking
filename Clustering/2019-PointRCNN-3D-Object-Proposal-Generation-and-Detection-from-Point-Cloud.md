
# PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud

- CVPR 2019

- [논문](https://arxiv.org/abs/1812.04244), [[Youtube] TF-KR PR-12 206](https://youtu.be/sFN_EgCsNzM?t=971)

- [code](https://paperswithcode.com/paper/pointrcnn-3d-object-proposal-generation-and)

Abstract - 제안 방식은 두 단계로 구성 되어 있음`In this paper, we propose PointRCNN for 3D object detection from raw point cloud. The whole framework is composed of two stages:`
- stage-1 for the bottom-up 3D proposal generation and 
- stage-2 for refining proposals in the canonical coordinates to obtain the final detection results. 

기존 방식 (RGB-D, BEV)와 달리 제안 방식의 1단계에서 배경과 물체를 세그멘테이션 한 후 **후보영역**을 생성 한다. `Instead of generating proposals from RGB image or projecting point cloud to bird’s view or voxels as previous methods do, our stage-1 sub-network directly generates a small number of high-quality 3D proposals from point cloud in a bottom-up manner via segmenting the point cloud of the whole scene into foreground points and background. `

제안 방식의 2단계 에서는 지역 특징을 좀더 잘 추출 하기 위해 후보 영역을 기준 좌표계로 변환 한다. `The stage-2 sub-network transforms the pooled points of each proposal to canonical coordinates to learn better local spatial features, which is combined with global semantic features of each point learned in stage-1 for accurate box refinement and confidence prediction. `

성능 평가 결과 좋음 `Extensive experiments on the 3D detection benchmark of KITTI dataset show that our proposed architecture outperforms state-of-the-art methods with remarkable margins by using only point cloud as input. `

> The code is available at https://github.com/sshaoshuai/PointRCNN.

## 1. Introduction

딥러닝 기반 2D 연구는 잘 되어 있지만, 3D는 아직 문제점이 있다. `Deep learning has achieved remarkable progress on 2D computer vision tasks, including object detection [8, 32, 16] and instance segmentation [6, 10, 20], etc. Beyond 2D scene understanding, 3D object detection is crucial and indispensable for many real-world applications, such as autonomous driving and domestic robots. While recent developed 2D detection algorithms are capable of handling large variations of viewpoints and background clutters in images, the detection of 3D objects with point clouds still faces great challenges from the irregular data format and large search space of 6 Degrees-of-Freedom (DoF) of 3D object.`

점군 기반 3D 물체 탐지의 문제점은 대부분 점군의 불류칙성에 기반 한다. `In autonomous driving, the most commonly used 3D sensors are the LiDAR sensors, which generate 3D point clouds to capture the 3D structures of the scenes. The difficulty of point cloud-based 3D object detection mainly lies in irregularity of the point clouds. `

![](https://i.imgur.com/fwbNqPV.png)
```
Figure 1. Comparison with state-of-the-art methods. Instead of generating proposals from fused feature maps of bird’s view and front view [14], or RGB images [25], our method directly generates 3D proposals from raw point cloud in a bottom-up manner.
```

최신 기법들은 아래와 같다. `State-of-the-art 3D detection methods `
- either leverage the mature 2D detection frameworks by projecting the point clouds into bird’s view [14, 42, 17] (see Fig. 1 (a)), to the frontal view [4, 38], 
- or to the regular 3D voxels [34, 43], which are not optimal and suffer from information loss during the quantization.

> 불규칙성을 규칙성을 가지게 변경 하는것 2D 픽셀 변환 , 3D 복셀 변환  

3D 점군을 직접 사용 하는 방법이 PointNET을 통해 제안 되었다. 하지만 이 방식에서의 후보영역 추출은 RGB-D를 이용 하였다. `Instead of transforming point cloud to voxels or other regular data structures for feature learning, Qi et al. [26, 28] proposed PointNet for learning 3D representations directly from point cloud data for point cloud classification and segmentation. As shown in Fig. 1 (b), their follow-up work [25] applied PointNet in 3D object detection to estimate the 3D bounding boxes based on the cropped frustum point cloud from the 2D RGB detection results. However, the performance of the method heavily relies on the 2D detection performance and cannot take the advantages of 3D information for generating robust bounding box proposals.`

2D 물체 탐지와 다르게 3D 물체 탐지는 3D 바운딩 박스를 통해서 쉽게 할수 있다. 다시 말해, 3D 물체 탐지 결과는 3D 세그멘테이션 을 위한 mask 로 사용될수 있다. 2D 물체 탐지 에서는 세그멘테이션을 위한 **약한 슈퍼 비젼** 만을 제공한다. `Unlike object detection from 2D images, 3D objects in autonomous driving scenes are naturally and well separated by annotated 3D bounding boxes. In other words, the training data for 3D object detection directly provides the semantic masks for 3D object segmentation. This is a key difference between 3D detection and 2D detection training data. In 2D object detection, the bounding boxes could only provide weak supervisions for semantic segmentation [5].`

위 특징을 기반으로 2단계로 이루어진 PointRCNN 아이디어를 제안한다. `Based on this observation, we present a novel two-stage 3D object detection framework, named PointRCNN, which directly operates on 3D point clouds and achieves robust and accurate 3D detection performance (see Fig. 1 (c)). `

**첫 단계**에서는 3D 후보 영역을 생성 한다. `The proposed framework consists of two stages, the first stage aims at generating 3D bounding box proposal in a bottomup scheme. `
- 배경을 제거 하여 물체들만 세그멘테이션을 먼저 수행 한후 후보 박스를 생성 한다. `By utilizing 3D bounding boxes to generate ground-truth segmentation mask, the first stage segments foreground points and generates a small number of bounding box proposals from the segmented points simultaneously. `
- 이 방식은 전체 처리를 하는 이전 방식 대비 자원 소모를 줄일수 있다. `Such a strategy avoids using the large number of 3D anchor boxes in the whole 3D space as previous methods [43, 14, 4] do and saves much computation.`

**두번째 단계**에서는 표준 좌표계로의 변환을 수행 한다. `The second stage of PointRCNN conducts canonical 3D box refinement. `
- After the 3D proposals are generated, a point cloud region pooling operation is adopted to pool learned point representations from stage-1. 
- 기존 방식이 직접 글로벌 좌표계 예측을 하였다면, 제안 방식은 pooled되 점군을 **표준 좌표계로 변환**후 **특징 정보**와 **세그멘테이션 mask** 정보화 합친다. `Unlike existing 3D methods that directly estimate the global box coordinates, the pooled 3D points are transformed to the canonical coordinates and combined with the pooled point features as well as the segmentation mask from stage-1 for learning relative coordinate refinement. `
- 이 방식을 이용하면 1단계에서 생성한 정보를 모두 활용할수 있다. `This strategy fully utilizes all information provided by our robust stage-1 segmentation and proposal sub-network.`
- 좀더 효과 적인 좌표계 정렬을 위해서 **full bin-based 3D box regression loss** 를 적용 하였다. `To learn more effective coordinate refinements, we also propose the full bin-based 3D box regression loss for proposal generation and refinement, and the ablation experiments show that it converges faster and achieves higher recall than other 3D box regression loss.`

기여도 `Our contributions could be summarized into three-fold.`
- (1) We propose a novel bottom-up point cloud-based 3D bounding box** proposal generation algorithm**, which generates a small number of high-quality 3D proposals via **segmenting the point cloud into foreground objects and background**. The learned point representation from segmentation is not only good at proposal generation but is also helpful for the later box refinement. 
- (2) The proposed canonical 3D bounding box refinement takes advantages of our high recall box proposals generated from stage-1 and learns to predict box coordinates refinements in the canonical coordinates with robust bin-based losses. 
- (3) Our proposed 3D detection framework PointRCNN outperforms state-of-theart methods with remarkable margins and ranks first among all published works as of Nov. 16 2018 on the 3D detection test board of KITTI by using only point clouds as input.

## 2. Related Work

### 2.1 3D object detection from 2D images.

이미지에서 3D BBox를 추출 하는 연구 `There are existing works on estimating the 3D bounding box from images. `
- [24, 15] leveraged the geometry constraints between 3D and 2D bounding box to recover the 3D object pose. 
- [1, 44, 23] exploited the similarity between 3D objects and the CAD models. 
- Chen et al. [2, 3] formulated the 3D geometric information of objects as an energy function to score the predefined 3D boxes. 

깊이 정보 부재와 외형 정보의 영향을 많이 받아 효과가 좋지 않다. `These works can only generate coarse 3D detection results due to the lack of depth information and can be substantially affected by appearance variations.`


### 2.2 3D object detection from point clouds. 

3D 점군 기반 물체 탐지 기술들 `State-of-the-art 3D object detection methods proposed various ways to learn discriminative features from the sparse 3D point clouds.`
 - [4, 14, 42, 17, 41] projected point cloud to **bird’s view ** and utilized **2D CNN **to learn the point cloud features for 3D box generation. 
 - Song et al. [34] and Zhou et al. [43] grouped the points into **voxels** and used 3D CNN to learn the features of voxels to generate 3D boxes. 

> 위 두 방식은 quantization로 인한 정보 손실이 발생 한다. 3D CNN릉 처리 효율이 좋지 않다. `However, the bird’s view projection and voxelization suffer from information loss due to the data quantization, and the 3D CNN is both memory and computation inefficient. `

[25, 39] utilized mature 2D detectors to generate 2D proposals from images and reduced the size of 3D points in each cropped image regions. 

PointNet [26, 28] is then used to learn the point cloud features for 3D box estimation. 
- But the 2D image based proposal generation might fail on some challenging cases that could only be well observed from 3D space. 
- Such failures could not be recovered by the 3D box estimation step. 

In contrast, our bottom-to-up 3D proposal generation method directly generates robust 3D proposals from point clouds, which is both efficient and quantization free.


### 2.3 Learning point cloud representations. 

기존 복셀화나 멀티뷰 방식 대신 PointNET은 점군을 직접 사용함으로써 속도와 정화도를 향상 시켰다. 이후 연구인 PointNET++에서는 지역 특징을 고려 하여 성능을 더욱 향상 시켰다. `Instead of representing the point cloud as voxels [22, 33, 35] or multi-view formats [27, 36, 37], Qi et al. [26] presented the PointNet architecture to directly learn point features from raw point clouds, which greatly increases the speed and accuracies of point cloud classification and segmentation. The follow-up works [28, 12] further improve the extracted feature quality by considering the local structures in point clouds.` 

Our work extends the point-based feature extractors to 3D point cloud-based object detection, leading to a novel two-stage 3D detection framework, which directly generate 3D box proposals and detection results from raw point clouds.

## 3. PointRCNN for Point Cloud 3D Detection

![](https://i.imgur.com/oanCxH2.png)
```
Figure 2. The PointRCNN architecture for 3D object detection from point cloud. 
The whole network consists of two parts: 
- (a) for generating 3D proposals from raw point cloud in a bottom-up manner. 
- (b) for refining the 3D proposals in canonical coordinate
```

In this section, we present our proposed two-stage detection framework, PointRCNN, for detecting 3D objects from irregular point cloud. 

The overall structure is illustrated in Fig. 2, which consists of 
- the bottom-up 3D proposal generation stage and 
- the canonical bounding box refinement stage.

### 3.1. Bottom-up 3D proposal generation via point cloud segmentation

2D 물체 탐지는 원 스테이지 or 투 스테이지 기법으로 나누어 진다. `Existing 2D object detection methods could be classified into one-stage and two-stage methods, `
- where one-stage methods [19, 21, 31, 30, 29] are generally faster but directly estimate object bounding boxes without refinement, 
- while two-stage methods [10, 18, 32, 8] generate proposals firstly and further refine the proposals and confidences in a second stage. 

3D 공간의 방대함과 대이터의 특징으로 2D 방식을 3D에 바로 적용하는것은 어렵다. `However, direct extension of the two-stage methods from 2D to 3D is non-trivial due to the huge 3D search space and the irregular format of point clouds. `
- AVOD [14] places 80-100k anchor boxes in the 3D space and pool features for each anchor in multiple views for generating proposals. 
- FPointNet [25] generates 2D proposals from 2D images, and estimate 3D boxes based on the 3D points cropped from the 2D regions, which might miss difficult objects that could only be clearly observed from 3D space. 

제안 방식에서는 점군 클러스터링 방식을 이용하여 스테이지 1에서 후보 영역을 추출 한다. `We propose an accurate and robust 3D proposal generation algorithm as our stage-1 sub-network based on whole scene point cloud segmentation.`
- 겹치는 부분이 없으므로 물체 구분이 쉽다. `We observe that objects in 3D scenes are naturally separated without overlapping each other. `
- 3D BBox 좌표 정보로 세그 멘테이션 마스트를 얻을수 있다. 즉, 박스 안에 있는것은 배경이 아니다. `All 3D objects’ segmentation masks could be directly obtained by their 3D bounding box annotations, i.e., 3D points inside 3D boxes are considered as foreground points.`

후보영역 추출 기법을 제안 하였다. 포인간 특징을 학습하여 후보 영역을 생성 하여 배경과 구분 하였다. `We therefore propose to generate 3D proposals in a bottom-up manner. Specifically, we learn point-wise features to segment the raw point cloud and to generate 3D proposals from the segmented foreground points simultaneously. `

제안 방식으로 탐색 범위를 줄일수 있다. `Based on this bottom-up strategy, our method avoids using a large set of predefined 3D boxes in the 3D space and significantly constrains the search space for 3D proposal generation. `

성능 평가 결과도 좋다. `The experiments show that our proposed 3D box proposal method achieves significantly higher recall than 3D anchor-based proposal generation methods.`

#### A. Learning point cloud representations.

특징 정보 학습을 위해 PointNet++을 사용하였다. `To learn discriminative point-wise features for describing the raw point clouds, we utilize the PointNet++ [28] with multi-scale grouping as our backbone network. `

다른 방식들도 교체/적용 가능하다. `There are several other alternative point-cloud network structures, such as [26, 13] or VoxelNet [43] with sparse convolutions [9], which could also be adopted as our backbone network.`

#### B. Foreground point segmentation.

전경 포인트들은 물체의 좌표와 방향을 계산 하는데 중요한 힌트를 제공한다. `The foreground points provide rich information on predicting their associated objects’ locations and orientations. By learning to segment the foreground points, the point-cloud network is forced to capture contextual information for making accurate point-wise prediction, which is also beneficial for 3D box generation. `

전경 포인트에서 후보영역을 생성하는 기법을 제안 하였다. 즉, **전경 포인트 추출**과 **3D BBox 생성**이 동시에 수행 된다. `We design the bottom-up 3D proposal generation method to generate 3D box proposals directly from the foreground points, i.e., the foreground segmentation and 3D box proposal generation are performed simultaneously.`

백본(PointNET++)에서 제공된 특징 정보에 아래 정보를 추가 하였다. ` Given the point-wise features encoded by the backbone point cloud network, we append`
- **segmentation head**를 추가 하였다. 이를 이용하여 전경 Mask를 유추 할수 있음 ` one segmentation head for estimating the foreground mask and`
- **Regression head** 3D 후부 영역 추출할수 있음  `one box regression head for generating 3D proposals.`

세그멘테이션을 위한 GT 마스크 정보는 학습데이터에서 제공된다. `For point segmentation, the ground-truth segmentation mask is naturally provided by the 3D ground-truth boxes.`

배경과 물체의 점군의 수가 다른 **Imbalance 문제** 해결을 위해 **Focal Loss**를 활용 하였다. ` The number of foreground points is generally much smaller than that of the background points for a large-scale outdoor scene. Thus we use the focal loss [19] to handle the class imbalance problem as`

![](https://cdn.mathpix.com/snip/images/KoNzYyUFXcjyvoT0i-veXbYVfDK4V0zWRgDvJOdLp2c.original.fullsize.png)

During training point cloud segmentation, we keep the default settings αt = 0.25 and γ = 2 as the original paper.

#### C. Bin-based 3D bounding box generation.

**box regression head ** 는 전경 세그멘테이션과 3D 후보영역 생성을 위해서도 필요 하다. `As we mentioned above, a box regression head is also appended for simultaneously generating bottom-up 3D proposals with the foreground point segmentation.` 학습 단계에서는 전경에서 3D bbox위치를 추론하기 위해서 **box regression head** 만 필요로 한다.  `During training, we only require the box regression head to regress 3D bounding box locations from foreground points. ` 

Note that although boxes are not regressed from the background points, those points also provide supporting information for generating boxes because of the receptive field of the point-cloud network.

3D bbox는 7개의 파라미터로 구성된다. 후보 박스 생성을 위해서 ** bin-based regression losses**를 제안 하였다. `A 3D bounding box is represented as (x, y, z, h, w, l, θ) in the LiDAR coordinate system, where (x, y, z) is the object center location, (h, w, l) is the object size, and θ is the object orientation from the bird’s view. To constrain the generated 3D box proposals, we propose bin-based regression losses for estimating 3D bounding boxes of objects. `

![](https://i.imgur.com/C8XP14G.png)
```
Figure 3. Illustration of bin-based localization. 
The surrounding area along X and Z axes of each foreground point is split into a series of bins to locate the object center.
```

물체의 중앙값을 예측 하기 위하여 각 전경 포인트의 주변 영역을 x,y축을 기준으로 여러개의 BIN으로 나누었다. `For estimating center location of an object, as shown in Fig. 3, we split the surrounding area of each foreground point into a series of discrete bins along the X and Z axes.`

Specifically, we set a search range S for each X and Z axis of the current foreground point, and each 1D search range is divided into bins of uniform length δ to represent different object centers (x, z) on the X-Z plane. 

제안된 **bin-based classification** + **cross-entropy loss**이 **smooth L1 loss**보다 성능이 더 좋다. `We observe that using bin-based classification with cross-entropy loss for the X and Z axes instead of direct regression with smooth L1 loss results in more accurate and robust center localization. `

The localization loss for the X or Z axis consists of two terms, 
- one term for bin classification along each X and Z axis, 
- and the other term for residual regression within the classified bin. 

For the center location y along the vertical Y axis, we directly utilize smooth L1 loss for the regression since most objects’ y values are within a very small range. Using the L1 loss is enough for obtaining accurate y values. 

The localization targets could therefore be formulated as

![](https://i.imgur.com/248Pn12.png)
- where (x (p) , y(p) , z(p) ) is the coordinates of a foreground point of interest, 
- (x p , yp , zp ) is the center coordinates of its corresponding object , 
- bin(p) x and bin(p) z are ground-truth bin assignments along X and Z axis, res(p) x and res(p) z are the ground-truth residual for further location refinement within the assigned bin, 
- and C is the bin length for normalization.

방향과 크기 : The targets of orientation θ and size (h, w, l) estimation are similar to those in [25]. 
- We divide the orientation 2π into n bins, 
- and calculate the bin classification target bin(p) θ and residual regression target res(p) θ in the same way as x or z prediction. 

크기 : The object size (h, w, l) is directly regressed by calculating residual (res (p) h , res (p) w , res (p) l ) w.r.t. the average object size of each class in the entire training set.

**x, z, θ, ** 예측 단계에서 In the inference stage, for the bin-based predicted parameters, x, z, θ, 
- we first choose the **bin center** with the highest predicted confidence and add the predicted residual to obtain the refined parameters. 

For other directly regressed parameters, including y, h, w, and l, 
- we add the predicted residual to their initial values.

The overall 3D bounding box regression loss L_{reg} with different loss terms for training could then be formulated as

![](https://cdn.mathpix.com/snip/images/MDTWUa4fJw8H6iuDAJVwFpGBTmcsNU_4IhXMSbU8CrE.original.fullsize.png)

- where N_{pos} is the number of foreground points, 
- binc (p) u and res c (p) u are the predicted bin assignments and residuals of the foreground point p, 
- bin(p) u and res(p) u are the ground-truth targets calculated as above, 
- F_{cls} denotes the cross-entropy classification loss, and 
- F_{reg} denotes the smooth L1 loss.

불필요한 후보영역 제거를 위해`To remove the redundant proposals,`
- we conduct nonmaximum suppression (NMS) based on the oriented IoU from bird’s view to generate a small number of high-quality proposals. 

사용된 학습 파라미터 : For training, we use 0.85 as the bird’s view IoU threshold and after NMS we keep top 300 proposals for training the stage-2 sub-network. 

사용된 추론 파라미터 : For inference, we use oriented NMS with IoU threshold 0.8, and only top 100 proposals are kept for the refinement of stage-2 sub-network.

### 3.2. Point cloud region pooling

후보 박스 생성 후에, 위치와 방향정보의 정제작업을 수행 한다. 각 후보영역의 좀더 지역적 정보를 학습하기 위해 스테이지 1에서의 특징에서 추가 정보 추출 방법을 제안 하였다. `After obtaining 3D bounding box proposals, we aim at refining the box locations and orientations based on the previously generated box proposals. To learn more specific local features of each proposal, we propose to pool 3D points and their corresponding point features from stage-1 according to the location of each 3D proposal.`

후보 박스에 대하여 추가 정보 적용을 위해 새 박스를 생성하였다. `For each 3D box proposal, bi = (xi , yi , zi , hi , wi , li , θi), we slightly enlarge it to create a new 3D box b e i = (xi , yi , zi , hi + η, wi + η, li + η, θi) to encode the additional information from its context, where η is a constant value for enlarging the size of box.`

For each point p = (x (p) , y(p) , z(p) ), an inside/outside test is performed to determine whether the point p is inside the enlarged bounding box proposal b e i . If so, the point and its features would be kept for refining the box bi . The features associated with the inside point p include its 3D point coordinates (x (p) , y(p) , z(p) ) ∈ R 3 , its laser reflection intensity r (p) ∈ R, its predicted segmentation mask m(p) ∈ {0, 1} from stage-1, and the C-dimensional learned point feature representation f (p) ∈ R C from stage-1.

We include the segmentation mask m(p) to differentiate the predicted foreground/background points within the enlarged box b e i . The learned point feature f (p) encodes valuable information via learning for segmentation and proposal generation therefore are also included. We eliminate the proposals that have no inside points in the following stage.


### 3.3. Canonical 3D bounding box refinement

As illustrated in Fig. 2 (b), the pooled points and their associated features (see Sec. 3.2) for each proposal are fed to our stage-2 sub-network for refining the 3D box locations as well as the foreground object confidence.

#### A. Canonical transformation. 

To take advantages of our high-recall box proposals from stage-1 and to estimate only the residuals of the box parameters of proposals, we transform the pooled points belonging to each proposal to the canonical coordinate system of the corresponding 3D proposal. 

As shown in Fig. 4, the canonical coordinate system for one 3D proposal denotes that 
- (1) the origin is located at the center of the box proposal; 
- (2) the local X0 and Z 0 axes are approximately parallel to the ground plane with X0 pointing towards the head direction of proposal and the other Z 0 axis perpendicular to X0 ; 
- (3) the Y 0 axis remains the same as that of the LiDAR coordinate system. 

All pooled points’ coordinates p of the box proposal should be transformed to the canonical coordinate system as p˜ by proper rotation and translation. Using the proposed canonical coordinate system enables the box refinement stage to learn better local spatial features for each proposal.

#### B. Feature learning for box proposal refinement. 

As we mentioned in Sec. 3.2, the refinement sub-network combines both the transformed local spatial points (features) p˜ as well as their global semantic features f (p) from stage-1 for further box and confidence refinement. Although the canonical transformation enables robust local spatial features learning, it inevitably loses depth information of each object. For instance, the far-away objects generally have much fewer points than nearby objects because of the fixed angular scanning resolution of the LiDAR sensors. To compensate for the lost depth information, we include the distance to the sensor into the features of point p.

For each proposal, its associated points’ local spatial features p˜ and the extra features [r (p) , m(p) , d(p) ] are first concatenated and fed to several fully-connected layers to encode their local features to the same dimension of the global features f (p) . Then the local features and global features are concatenated and fed into a network following the structure of [28] to obtain a discriminative feature vector for the following confidence classification and box refinement.

#### C. Losses for box proposal refinement. 

We adopt the similar bin-based regression losses for proposal refinement. A ground-truth box is assigned to a 3D box proposal for learning box refinement if their 3D IoU is greater than 0.55. Both the 3D proposals and their corresponding 3D ground-truth boxes are transformed into the canonical coordinate systems, which means the 3D proposal bi = (xi , yi , zi , hi , wi , li , θi) and 3D ground-truth box b gt i = (x gt i , y gt i , z gt i , hgt i , w gt i , lgt i , θgt i ) would be transformed to

![](https://i.imgur.com/KK9JSPV.png)

The training targets for the ith box proposal’s center location, (bini ∆x , bini ∆z , resi ∆x , resi ∆z , resi ∆y ), are set in the same way as Eq. (2) except that we use smaller search range S for refining the locations of 3D proposals. We still directly regress size residual (resi ∆h , resi ∆w, resi ∆l ) w.r.t. the average object size of each class in the training set since the pooled sparse points usually could not provide enough information of the proposal size (hi , wi , li).

For refining the orientation, we assume that the angular difference w.r.t. the ground-truth orientation, θ gt i − θi , is within the range [− π 4 , π 4 ], based on the fact that the 3D IoU between a proposal and their ground-truth box is at least 0.55. Therefore, we divide π 2 into discrete bins with the bin size ω and predict the bin-based orientation targets as 

![](https://cdn.mathpix.com/snip/images/QMKEvTm2zJ4LDaX3hOUCIkRYcHnehaK3VyCoCMievyY.original.fullsize.png)

Therefore, the overall loss for the stage-2 sub-network can be formulated as
![](https://i.imgur.com/Fd6YQ3a.png)

where B is the set of 3D proposals from stage-1 and Bpos stores the positive proposals for regression, probi is the estimated confidence of b˜ i and labeli is the corresponding label, Fcls is the cross entropy loss to supervise the predicted confidence, L˜ (i) bin and L˜ (i) res are similar to L (p) bin and L (p) res in Eq. (3) with the new targets calculated by b˜i and b˜gt i as above.

We finally apply oriented NMS with bird’s view IoU threshold 0.01 to remove the overlapping bounding boxes and generate the 3D bounding boxes for detected objects.

## 4. Experiments

PointRCNN is evaluated on the challenging 3D object detection benchmark of KITTI dataset [7]. 
- We first introduce the implementation details of PointRCNN in Sec. 4.1. 
- In Sec. 4.2, we perform a comparison with state-of-the-art 3D detection methods. 
- Finally, we conduct extensive ablation studies to analyze PointRCNN in Sec. 4.3.

### 4.1. Implementation Details

#### A. Network Architecture. 

For each 3D point-cloud scene in the training set, we subsample 16,384 points from each scene as the inputs. 
- For scenes with the number of points fewer than 16,384, we randomly repeat the points to obtain 16,384 points. 

For the stage-1 sub-network, we follow the network structure of [28-PointNet++], where four set-abstraction layers with multi-scale grouping are used to subsample points into groups with sizes 4096, 1024, 256, 64. 

Four feature propagation layers are then used to obtain the per-point feature vectors for segmentation and proposal generation. 

For the box proposal refinement sub-network, we randomly sample 512 points from the pooled region of each proposal as the input of the refinement sub-network. 

Three set abstraction layers with single-scale grouping [28] (with group sizes 128, 32, 1) are used to generate a single feature vector for object confidence classification and proposal location refinement.

#### B. The training scheme. 

Here we report the training details of car category since it has the majority of samples in the KITTI dataset, and the proposed method could be extended to other categories (like pedestrian and cyclist) easily with little modifications of hyper parameters. For stage-1 sub-network, all points inside the 3D groundtruth boxes are considered as foreground points and others points are treated as background points. During training, we ignore background points near the object boundaries by enlarging the 3D ground-truth boxes by 0.2m on each side of object for robust segmentation since the 3D groundtruth boxes may have small variations. For the bin-based proposal generation, the hyper parameters are set as search range S = 3m, bin size δ = 0.5m and orientation bin number n = 12.

To train the stage-2 sub-network, we randomly augment the 3D proposals with small variations to increase the diversity of proposals. For training the box classification head, a proposal is considered as positive if its maximum 3D IoU with ground-truth boxes is above 0.6, and is treated as negative if its maximum 3D IoU is below 0.45. We use 3D IoU 0.55 as the minimum threshold of proposals for the training of box regression head. For the bin-based proposal refinement, search range is S = 1.5m, localization bin size is δ = 0.5m and orientation bin size is ω = 10◦ . The context length of point cloud pooling is η = 1.0m

The two stage sub-networks of PointRCNN are trained separately. The stage-1 sub-network is trained for 200 epochs with batch size 16 and learning rate 0.002, while the stage-2 sub-network is trained for 50 epochs with batch size 256 and learning rate 0.002. During training, we conduct data augmentation of random flip, scaling with a scale factor sampled from [0.95, 1.05] and rotation around vertical Y axis between [-10, 10] degrees. Inspired by [40], to simulate objects with various environments, we also put several new ground-truth boxes and their inside points from other scenes to the same locations of current training scene by randomly selecting non-overlapping boxes, and this augmentation is denoted as GT-AUG in the following sections.


> 후략 

## 5. Conclusion 

We have presented PointRCNN, a novel 3D object detector for detecting 3D objects from raw point cloud. The proposed stage-1 network directly generates 3D proposals from point cloud in a bottom-up manner, which achieves significantly higher recall than previous proposal generation methods. The stage-2 network refines the proposals in the canonical coordinate by combining semantic features and local spatial features. Moreover, the newly proposed binbased loss has demonstrated its efficiency and effectiveness for 3D bounding box regression. The experiments show that PointRCNN outperforms previous state-of-the-art methods with remarkable margins on the challenging 3D detection benchmark of KITTI dataset.
