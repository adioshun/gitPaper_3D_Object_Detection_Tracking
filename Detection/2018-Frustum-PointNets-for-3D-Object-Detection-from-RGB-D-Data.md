
# [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf)

> [홈페이지](http://stanford.edu/~rqi/frustum-pointnets/), [깃허브](https://github.com/charlesq34/frustum-pointnets), [Article](https://blog.csdn.net/shuqiaos/article/details/82752100), [Docker](https://hub.docker.com/r/luca911014/frustum-pointnet)

In this work, we study 3D object detection from RGBD data in both indoor and outdoor scenes. While previous methods focus on images or 3D voxels, often obscuring natural 3D patterns and invariances of 3D data, we directly operate on raw point clouds by popping up RGB-D scans. However, a key challenge of this approach is how to efficiently localize objects in point clouds of large-scale scenes (region proposal). Instead of solely relying on 3D proposals, our method leverages both mature 2D object detectors and advanced 3D deep learning for object localization, achieving efficiency as well as high recall for even small objects. Benefited from learning directly in raw point clouds, our method is also able to precisely estimate 3D bounding boxes even under strong occlusion or with very sparse points. Evaluated on KITTI and SUN RGB-D 3D detection benchmarks, our method outperforms the state of the art by remarkable margins while having real-time capability.



## 1. Introduction

본 논문에서는 3D 인지 시스템에 대하여 다루고 있다. (탐지, 분류, 방향) `Recently, great progress has been made on 2D image understanding tasks, such as object detection [13] and instance segmentation [14]. However, beyond getting 2D bounding boxes or pixel masks, 3D understanding is eagerly in demand in many applications such as autonomous driving and augmented reality (AR). With the popularity of 3D sensors deployed on mobile devices and autonomous vehicles, more and more 3D data is captured and processed. In this work, we study one of the most important 3D perception tasks – 3D object detection, which classifies the object category and estimates oriented 3D bounding boxes of physical objects from 3D sensor data.`

3D 센서 데이터가 포인트 클라우드 형태로 저장 되지만 이를 분석 하는 기술은 아직 연구 분야 이다. 기존에는 포인트 클라우드를 **이미지로 투영**하거나 **volumetric grids by quantization**로 만들어 CNN에 적용 하였다. `While 3D sensor data is often in the form of point clouds, how to represent point cloud and what deep net architectures to use for 3D object detection remains an open problem. Most existing works convert 3D point clouds to images by projection [36, 26] or to volumetric grids by quantization [40, 23, 26] and then apply convolutional networks.`

**PointNets**에서는 위와 같은 변경 과정 없이 포인트 클라우드를 바로 사용하여 좋은 성과를 보였다. `This data representation transformation, however, may obscure natural 3D patterns and invariances of the data. Recently, a number of papers have proposed to process point clouds directly without converting them to other formats. For example, [25, 27] proposed new types of deep net architectures, called PointNets, which have shown superior performance and efficiency in several 3D understanding tasks such as object classification and semantic segmentation.`

포인트넷이 분류 기능은 좋지만 탐색은 별도의 문제이다. 이를 위해서 후보 영역 추천 하는 기능이 있어야한다. 2D의 기술을 활용하여 슬라이딩 윈도우 기법을 쓰거나 3D Region Proposal Network를 사용할수도 있다. 하지만 이들은 컴퓨팅 부하가 크다. `While PointNets are capable of classifying a whole point cloud or predicting a semantic class for each point in a point cloud, it is unclear how this architecture can be used for instance-level 3D object detection. Towards this goal, we have to address one key challenge: how to efficiently propose possible locations of 3D objects in a 3D space. Imitating the practice in image detection, it is straightforward to enumerate candidate 3D boxes by sliding windows [8] or by 3D region proposal networks such as [33]. However, the computational complexity of 3D search typically grows cubically with respect to resolution and becomes too expensive for large scenes or real-time applications such as autonomous driving.`

![](https://i.imgur.com/r5ipOrn.png)

본 논문에서는 차원 축소를 통해서 탐색 범위를 줄였다. 2D 이미지 기반 탐지기의 기능을 활용 하였다. `Instead, in this work, we reduce the search space following the dimension reduction principle: we take the advantage of mature 2D object detectors (Fig. 1).`
- First, we extract the 3D bounding frustum of an object by extruding 2D bounding boxes from image detectors. 
- Then, within the 3D space trimmed by each of the 3D frustums, we consecutively perform **3D object instance segmentation** and **amodal 3D bounding box regression** using two variants of **PointNet**. 
	- The segmentation network predicts the 3D mask of the object of interest (i.e. instance segmentation); 
	- and the regression network estimates the amodal 3D bounding box (covering the entire object even if only part of it is visible).

이전방식에서도 RGB-D의 2D map을 CNN에 적용하는 방식과는 반대로 제안 방식은 좀더 **3D-centric** 하다. 제안 방식은 Depth map을 3D 포인트클라우드 처럼 사용하였다. 이러한 접근은 아래와 같은 장점이 있다. `In contrast to previous work that treats RGB-D data as 2D maps for CNNs, our method is more 3D-centric as we lift depth maps to 3D point clouds and process them using 3D tools. This 3D-centric view enables new capabilities for exploring 3D data in a more effective manner. `
- First, in our pipeline, a few transformations are applied successively on 3D coordinates, which align point clouds into a sequence of more constrained and canonical frames. These alignments factor out pose variations in data, and thus make 3D geometry pattern more evident, leading to an easier job of 3D learners. 
- Second, learning in 3D space can better exploits the geometric and topological structure of 3D space. In principle, all objects live in 3D space; therefore, we believe that many geometric structures, such as repetition, planarity, and symmetry, are more naturally parameterized and captured by learners that directly operate in 3D space. The usefulness of this 3D-centric network design philosophy has been supported by much recent experimental evidence.

제안 기술은 성능적으로도 좋다. `Our method achieve leading positions on KITTI 3D object detection [1] and bird’s eye view detection [2] benchmarks. Compared with the previous state of the art [6], our method is 8.04% better on 3D car AP with high efficiency (running at 5 fps). Our method also fits well to indoor RGBD data where we have achieved 8.9% and 6.4% better 3D mAP than [16] and [30] on SUN-RGBD while running one to three orders of magnitude faster`

논문의 기여도는 아래와 같다. `The key contributions of our work are as follows:`
- We propose a novel framework for RGB-D data based 3D object detection called Frustum PointNets.
- We show how we can train 3D object detectors under our framework and achieve state-of-the-art performance on standard 3D object detection benchmarks. 
- We provide extensive quantitative evaluations to validate our design choices as well as rich qualitative results for understanding the strengths and limitations of our method.

## 2. Related Work

### 2.1 3D Object Detection from RGB-D Data

Researchers have approached the 3D detection problem by taking various ways to represent RGB-D data.

#### A. Front view image based methods

- [4, 24, 41] take monocular RGB images and shape priors or occlusion patterns to infer 3D bounding boxes. 

- [18, 7] represent depth data as 2D maps and apply CNNs to localize objects in 2D image. 

- In comparison we represent depth as a point cloud and use advanced 3D deep networks (PointNets) that can exploit 3D geometry more effectively.

#### B. Bird’s eye view based methods

- MV3D는 BEV를 이용하여 RPN을 학습 시켰다. `MV3D [6] projects LiDAR point cloud to bird’s eye view and trains a region proposal network (RPN [29]) for 3D bounding box proposal. `

- 작은 물체는 탐지 못하고, 버티컬로 있는 여러 물체를 탐지 하기 어렵다. `However, the method lags behind in detecting small objects, such as pedestrians and cyclists and cannot easily adapt to scenes with multiple objects in vertical direction.`

#### C. 3D based methods

- [38, 34] train 3D object classifiers by **SVMs** on hand-designed geometry features extracted from point cloud and then localize objects using **sliding window** search. 

- [8] extends [38] by replacing SVM with 3D CNN on voxelized 3D grids. 

- [30] designs new geometric features for 3D object detection in a point cloud. 

- [35, 17] convert a point cloud of the entire scene into a volumetric grid and use 3D volumetric CNN for object proposal and classification. 

- 위 방식들은 3D-CNN 연산이나 3D 탐색 범위로 인해 계산 부하가 크다. `Computation cost for those method is usually quite high due to the expensive cost of 3D convolutions and large 3D search space. `

- Recently, [16] proposes a 2D driven 3D object detection method that is similar to ours in spirit. 
	- However, they use hand-crafted features (based on histogram of point coordinates) with simple fully connected networks to regress 3D box location and pose, which is sub-optimal in both speed and performance. 
	- In contrast, we propose a more flexible and effective solution with deep 3D feature learning (PointNets).

### 2.2 Deep Learning on Point Clouds

- Most existing works convert point clouds to **images** or **volumetric** forms before feature learning. 

- [40, 23, 26] voxelize point clouds into volumetric grids and generalize image CNNs to 3D CNNs. 

- [19, 31, 39, 8] design more efficient 3D CNN or neural network architectures that exploit sparsity in point cloud. 

- 그러나 위 방식들은 **복셀화**가 필수 적이다. `However, these CNN based methods still require quantitization of point clouds with certain voxel resolution. `

- 최근에는 (복셀화 없이) 직접 포인트 클라우드를 사용하는 방법이 제안되었다. : PointNet `Recently, a few works [25, 27] propose a novel type of network architectures (PointNets) that directly consumes raw point clouds without converting them to other formats. `

- PointNet이 분류나 세그멘테이션에 특화 되어 있는 반면 제안 방식은 탐지도 같이 수행 한다. `While PointNets have been applied to single object classification and semantic segmentation, our work explores how to extend the architecture for the purpose of 3D object detection.`

## 3. Problem Definition

제안 방식의 목표는 주어진 RGB-D데이터를 기반으로 분류와 물체위 위치를 찾는것이다. 깊이 정보는 **RGB-카메라 좌표계**의 포인트 클라우드로 표현된다. 투영 매트릭스 정보는 알고 있으므로 2D 이미지 영역에서 3D **frustum**을 알수 있다. `Given RGB-D data as input, our goal is to classify and localize objects in 3D space. The depth data, obtained from LiDAR or indoor depth sensors, is represented as a point cloud in RGB camera coordinates. The projection matrix is also known so that we can get a 3D frustum from a 2D image region.`


각 물체는 클래스 정보와 바운딩 박스로 표현된다. 가려져 있어도 바운딩 박스 처리가 가능하다.  Each object is represented by a class (one among k predefined classes) and an amodal 3D bounding box. The amodal box bounds the complete object even if part of the object is occluded or truncated. The 3D box is parameterized by its size h, w, l, center cx, cy, cz, and orientation θ, φ, ψ relative to a predefined canonical pose for each category. In our implementation, we only consider the heading angle θ around the up-axis for orientation.

## 4. 3D Detection with Frustum PointNets

![](https://i.imgur.com/ETXxOIA.png)

제안 방식의 3가지 모듈 `As shown in Fig. 2, our system for 3D object detection consists of three modules:`
-  frustum proposal, 
- 3D instance segmentation, and 
- 3D amodal bounding box estimation. 

We will introduce each module in the following subsections. We will focus on the pipeline and functionality of each module, and refer readers to supplementary for specific architectures of the deep networks involved.

### 4.1 Frustum Proposal

3D데이터는 해상도가 좋지 않기에 2D 이미지데이터를 이용하여 **물체 탐지**와 **분류** 를 수행 하였다. `The resolution of data produced by most 3D sensors, especially real-time depth sensors, is still lower than RGB images from commodity cameras. Therefore, we leverage mature 2D object detector to propose 2D object regions in RGB images as well as to classify objects.`

![](https://i.imgur.com/t9VV5DT.png)

2D 물체 탐지로 추출된 3D 포인트 클라우드는 frustum형태이다. 위 그림에서 보듯이 frustum의 방향은 여러 곳일수 있다. 따라서 Normalization이 필요 하다. 본 논문에서는 이미지 평면과 수직이도록 frustum의 중심 axis를 회전 시켰다. 이러한 normalization을 통해 알고리즘이 ** rotation-invariance** 할수 있다. `With a known camera projection matrix, a 2D bounding box can be lifted to a frustum (with near and far planes specified by depth sensor range) that defines a 3D search space for the object. We then collect all points within the frustum to form a frustum point cloud. As shown in Fig 4 (a), frustums may orient towards many different directions, which result in large variation in the placement of point clouds. We therefore normalize the frustums by rotating them toward a center view such that the center axis of the frustum is orthogonal to the image plane. This normalization helps improve the rotation-invariance of the algorithm. We call this entire procedure for extracting frustum point clouds from RGB-D data frustum proposal generation.`

> normalize  방법 : rotating them toward a center view (=orthogonal to the image plane)

2D 물체 탐지기의 학습 방법 설명 들 `While our 3D detection framework is agnostic to the exact method for 2D region proposal, we adopt a FPN [20] based model. We pre-train the model weights on ImageNet classification and COCO object detection datasets and further fine-tune it on a KITTI 2D object detection dataset to classify and predict amodal 2D boxes. More details of the 2D detector training are provided in the supplementary.`

### 4.2. 3D Instance Segmentation

2D이미지에서 3D 물체 Location을 찾는 방법들 `Given a 2D image region (and its corresponding 3D frustum), several methods might be used to obtain 3D location of the object:`
- One straightforward solution is to directly **regress 3D object locations** (e.g., by 3D bounding box) from a depth map using 2D CNNs. 
	- 단점 : 가려져 있거나, 배경이 있는 환경에서 적합하지 않음 `However, this problem is not easy as occluding objects and background clutter is common in natural scenes (as in Fig. 3), which may severely distract the 3D localization task.`

물체는 3D 공간상에 분포 되어 있으므로 이미지처럼 바로 옆에 있는것으로 표현되어도 상대적 거리가 있어 3D상 세그멘테이션은 더 쉽다. **포인트넷**을 이용하여 세그멘테이션을 수행 하였다. `Because objects are naturally separated in physical space, segmentation in 3D point cloud is much more natural and easier than that in images where pixels from distant objects can be near-by to each other. Having observed this fact, we propose to segment instances in 3D point cloud instead of in 2D image or depth map. Similar to Mask-RCNN [14], which achieves instance segmentation by binary classification of pixels in image regions, we realize 3D instance segmentation using a PointNet-based network on point clouds in frustums.`


인스턴스 세그멘테이션을 통해 **residual based 3D localization**이 가능하다. 이 방식이 **egressing the absolute 3D location** 하는것보다 원거리 물체에 대한 불확실성이 적어 더 좋다. 이후 그림Fig. 4 (c)처럼  **3D mask coordinates**상의 바운딩 박스 중심값을 예측 하였다. `Based on 3D instance segmentation, we are able to achieve residual based 3D localization. That is, rather than regressing the absolute 3D location of the object whose offset from the sensor may vary in large ranges (e.g. from 5m to beyond 50m in KITTI data), we predict the 3D bounding box center in a local coordinate system – 3D mask coordinates as shown in Fig. 4 (c).`


#### 3D Instance Segmentation PointNet.

네트워크는 frustum형태의 포인트 클라우드를 입력으로 받아 얼마나 **관심 물체**에 속하는지 확률값을 출력한다. 하나의 frustum에는 하나의 물체만 속해 있다. 여기서 다른 포인트들이란 비관심 물체 뒤에 있거나, 가려진 물체, 관련없는 영역(바닥)이다. 2D 인스턴스 세그멘테이션과 같이 frustum의 위치에 따라서 frustum상 물체의 포인트는 잘려져 있거나 가려져 있을수 있다. 따라서 이런상황도 대처 할수 있게 하였다. `The network takes a point cloud in frustum and predicts a probability score for each point that indicates how likely the point belongs to the object of interest. Note that each frustum contains exactly one object of interest. Here those “other” points could be points of non-relevant areas (such as ground, vegetation) or other instances that occlude or are behind the object of interest. Similar to the case in 2D instance segmentation, depending on the position of the frustum, object points in one frustum may become cluttered or occlude points in another. Therefore, our segmentation PointNet is learning the occlusion and clutter patterns as well as recognizing the geometry for the object of a certain category.`

다중 분류 문제에서도 2D 탐지기를 활용 하였다. 예를 들어 관심 물체가 보행자라는것을 알고 있다면 이 사전 정보를 사람과 비슷한 geometries를 찾는데 사용한다. `In a multi-class detection case, we also leverage the semantics from a 2D detector for better instance segmentation. For example, if we know the object of interest is a pedestrian, then the segmentation network can use this prior to find geometries that look like a person.`

Specifically, in our architecture we encode the semantic category as a one-hot class vector (k dimensional for the pre-defined k categories) and concatenate the one-hot vector to the intermediate point cloud features. More details of the specific architectures are described in the supplementary.

인스턴스 세그멘테이션 이후에는 관심 물체라고 생각되는 포인트들 **masking**에서 추출 된다. 추출된 포인트 클라우드는 이후 **translational**에 강건한 알고리즘을 위해  좌료를 노멀라이즈 한다. `After 3D instance segmentation, points that are classified as the object of interest are extracted (“masking” in Fig. 2). Having obtained these segmented object points, we further normalize its coordinates to boost the translational invariance of the algorithm, following the same rationale as in the frustum proposal step.`

본 구현물에서는 물체의 중앙을 기준으로 x,y,z의 좌표계를 변경한다.  중요한점은 고의적으로 크기를 조정하지 않는것이다. 왜냐 하면 보는 방향에 따라서  bounding sphere 크기가 영향을 받기 때문이다. 그리고 실제 포인트의 크기 정보는 박스 크기 예측에 도움을 주기 때문이다. `In our implementation, we transform the point cloud into a local coordinate by subtracting XYZ values by its centroid. This is illustrated in Fig. 4 (c). Note that we intentionally do not scale the point cloud, because the bounding sphere size of a partial point cloud can be greatly affected by viewpoints and the real size of the point cloud helps the box size estimation.`

실험결과 좌표 변환이나 frustum rotation은 3D 물체 탐지에 크리티컬한 요소였다. `In our experiments, we find that coordinate transformations such as the one above and the previous frustum rotation are critical for 3D detection result as shown in Tab. 8.`

### 4.3. Amodal 3D Box Estimation 

본 모듈에서는 획득한 물체에 대해 회전 방향 바운딩 박스를 **box regression PointNet** + **preprocessing transformer network**를 통해 도출 한다.  `Given the segmented object points (in 3D mask coordinate), this module estimates the object’s amodal oriented 3D bounding box by using a box regression PointNet together with a preprocessing transformer network.`

#### A. Learning-based 3D Alignment by T-Net 

비록 각 물체의 중심점에따라 aligned된 물체를 획득 하여도 mask coordinate frame의origin은   amodal box center과는 차이가 크다. `Even though we have aligned segmented object points according to their centroid position, we find that the origin of the mask coordinate frame (Fig. 4 (c)) may still be quite far from the amodal box center.`

따라서 본 논문에서는  **light-weight regression PointNet (T-Net)**를 사용하여서 물체의 진짜 중앙을 찾고 좌표계를 변환 하였다.  즉, 예측된 중앙이 origin이 되게 한것이다. 사용된 T-Net은 [25]와 비슷하며 STN[15]의 일종이다. 다른점은 기존은 비 지도 방식인데 우리는 지도 방식이다. `  We therefore propose to use a light-weight regression PointNet (T-Net) to estimate the true center of the complete object and then transform the coordinate such that the predicted center becomes the origin (Fig. 4 (d)). The architecture and training of our T-Net is similar to the T-Net in [25], which can be thought of as a special type of spatial transformer network (STN) [15]. However, different from the original STN that has no direct supervision on transformation, we explicitly supervise our translation network to predict center residuals from the mask coordinate origin to real object center.`

```
[25] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017.

```

#### B. Amodal 3D Box Estimation PointNet 

이 네트워크는 바운딩박스의 3D 좌표를 출력한다. 네트워크 구조는 분류기와 비슷 하지만 산출물이 바운딩 박스 좌표 이다. `The box estimation network predicts amodal bounding boxes (for entire object even if part of it is unseen) for objects given an object point cloud in 3D object coordinate (Fig. 4 (d)). The network architecture is similar to that for object classification [25, 27], however the output is no longer object class scores but parameters for a 3D bounding box.`

3D bounding box는 아래 값으로 정의 된다. `As stated in Sec. 3, we parameterize a 3D bounding box by its center (cx, cy, cz), size (h, w, l) and heading angle θ (along up-axis). `
- center (cx, cy, cz)
- size (h, w, l) 
- heading angle θsms 

##### 가. Center

박스의 중앙을 예측할때 **residual** 접근법을 사용하였다. box estimation networ를 통해 도출된 **center residual**은 T-Net을 통한 이전 **center residual** 와  **the masked points’ centroid** 와 결합되어  절대 중앙값을 산출 한다. `We take a “residual” approach for box center estimation. The center residual predicted by the box estimation network is combined with the previous center residual from the T-Net and the masked points’ centroid to recover an absolute center (Eq. 1). `

$$ C_{pred} = C_{mask} + ∆C_{t−net} + ∆C_{box−net} $$

![](https://i.imgur.com/FeqtMzQ.png)


#### 나. Size 

박스 크기와 방향계산을 위해서는 이전 [29,24]연구를 따르고,  여기에 분류와 회귀공식을 혼합하였다. `For box size and heading angle, we follow previous works [29, 24] and use a hybrid of classification and regression formulations.`
-  Specifically we pre-define NS size templates and NH equally split angle bins. 

제안 모델의 결과물은 아래와 같다. `Our model will both`
- classify size/heading (NS scores for size, NH scores for heading) to those pre-defined categories 
- as well as predict residual numbers for each category (3×NS residual dimensions for height, width, length, NH residual angles for heading). 

최종 결과물 보습 : In the end the net outputs 3 + 4 × NS + 2 × NH numbers in total.

### 4.4. Training with Multi-task Losses

제안 방식은 아래 공식의 **multi-task losses**를 이용하여서 반복적으로 3개의 네트워크를 최적화 한다. `We simultaneously optimize the three nets involved (3D instance segmentation PointNet, T-Net and amodal box estimation PointNet) with multi-task losses (as in Eq. 2). `

![](https://i.imgur.com/xitssUX.png)
- Lc1−reg is for T-Net and Lc2−reg is for center regression of box estimation net. 
- Lh−cls and Lh−reg are losses for heading angle prediction 
- while Ls−cls and Ls−reg are for box size. 
- Softmax is used for all classification tasks and smooth-l1 (huber) loss is used for all regression cases.

#### Corner Loss for Joint Optimization of Box Parameters

While our 3D bounding box parameterization is compact and complete, learning is not optimized for final 3D box accuracy – center, size and heading have separate loss terms. 

Imagine cases where center and size are accurately predicted but heading angle is off – the 3D IoU with ground truth box will then be dominated by the angle error. Ideally all three terms (center,size,heading) should be jointly optimized for best 3D box estimation (under IoU metric).

To resolve this problem we propose a novel regularization loss, the **corner loss**:

![](https://i.imgur.com/XdjjeRf.png)

기본적으로 **Corner loss**는 예측된 8개 코너와 GT의 8개 코너의 거리의 합이다. `In essence, the corner loss is the sum of the distances between the eight corners of a predicted box and a ground truth box. `

코너의 위치는 중앙값, 크기, 방향의 정보로 정해 지기 때문에 **corner loss**는 multi-task training을 regularize할수 있다. `Since corner positions are jointly determined by center, size and heading, the corner loss is able to regularize the multi-task training for those parameters.`

계산 방법 `To compute the corner loss,`
- we firstly construct NS × NH “anchor” boxes from all size templates and heading angle bins. 
- The anchor boxes are then translated to the estimated box center. 
- We denote the anchor box corners as P ij k , where i, j, k are indices for the size class, heading class, and (predefined) corner order, respectively. 
- To avoid large penalty from flipped heading estimation, 
	- we further compute distances to corners (P ∗∗ k ) from the flipped ground truth box and use the minimum of the original and flipped cases. 
	- δij , which is one for the ground truth size/heading class and zero else wise, is a two-dimensional mask used to select the distance term we care about.

## 5. Experiments


---  
## 별첨 자료 

### Appendix A. Overview

This document provides additional technical details, extra analysis experiments, more quantitative results and qualitative test results to the main paper. 
- In Sec.B we provide more details on network architectures of PointNets and training parameters while 
- Sec. C explains more about our 2D detector. 
- Sec. D shows how our framework can be extended to bird’s eye view (BV) proposals and how combining BV and RGB proposals can further improve detection performance. 
- Then Sec. E presents results from more analysis experiments. 
- At last, Sec. F shows more visualization results for 3D detection on SUN-RGBD dataset.

### Appendix B. Details on Frustum PointNets (Sec 4.2, 4.3)

#### B.1. Network Architectures

포인넷을 활용하였다. 차이점은 **class one-hot vector **을 위한 링크를 추가 한것이다. 이 추가점은 RGB이미지를 이용하여서 인스턴스 세그멘테이션과 바운딩박스 추정을 가능하게 한다. `We adopt similar network architectures as in the original works of PointNet [25] and PointNet++ [27] for our v1 and v2 models respectively. What is different is that we add an extra link for class one-hot vector such that instance segmentation and bounding box estimation can leverage semantics predicted from RGB images. The detailed network architectures are shown in Fig. 8. `


![](https://i.imgur.com/1OJBDBj.png)
```python 
v1 models are based on PointNet [25].
''' 3D instance segmentation PointNet v1 network.
Input:
    point_cloud: TF tensor in shape (B,N,4)
        frustum point clouds with XYZ and intensity in point channels
        XYZs are in frustum coordinate
    one_hot_vec: TF tensor in shape (B,3)
        length-3 vectors indicating predicted object type
    is_training: TF boolean scalar
    bn_decay: TF float scalar
    end_points: dict
Output:
    logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
    end_points: dict
'''
```

For v1 model our architecture involves 
- point embedding layers (as shared MLP on each point independently), 
- a max pooling layer and 
- per-point classification multi-layer perceptron (MLP) based on aggregated information from global feature and each point as well as an one-hot class vector. 

Note that we do not use the transformer networks as in [25] because frustum points are viewpoint based (not complete point cloud as in [25]) and are already normalized by frustum rotation. 

In addition to XYZ , we also leverage LiDAR intensity as a fourth channel. 




![](https://i.imgur.com/wAvgheC.png)
```
v2 models are based on PointNet++ [27] set abstraction (SA) and feature propagation (FP) layers.
```

For v2 model we use 
- set abstraction layers for hierarchical feature learning in point clouds. 


LIDAR의 속성상 원거리 포인트는 방사되기 때문에 이를 해결(density variations.)하기 위한 MSG레이어를 사용하였다. `In addition, because LiDAR point cloud gets increasingly sparse as it gets farther, feature learning has to be robust to those density variations. Therefore we used a robust type of set abstraction layers – multi-scale grouping (MSG) layers as introduced in [27] for the segmentation network. With hierarchical features and learned robustness to varying densities, our v2 model shows superior performance than v1 model in both segmentation and box estimation.`


#### B.2. Data Augmentation and Training

##### Data augmentation 

오버피팅을 막기 위해 두 종류의 **augmentation**을 하였다.` Data augmentation plays an important role in preventing model overfitting. Our augmentation involves two branches: one is 2D box augmentation and the other is frustum point cloud augmentation. `
- 2D box augmentation 
- frustum point cloud augmentation

바운딩 박스 증폭(augmentation)은 거리 박스의 크기를 다양하게 변경 시키는 것이다. `We use ground truth 2D boxes to generate frustum point clouds for Frustum PointNets training and augment the 2D boxes by random translation and scaling. Specifically, we firstly compute the 2D box height (h) and width (w) and translate the 2D box center by random distances sampled from Uniform[−0.1w, 0.1w] and Uniform[−0.1h, 0.1h] in u,v directions respectively. The height and width are also augmented by two random scaling factor sampled from Uniform[0.9, 1.1].`

포인트 클라우드 증폭은 세가지 방식으로 진행 된다. `We augment each frustum point cloud by three ways.`
- First, we randomly sample a subset of points from the frustum point cloud on the fly (1,024 for KITTI and 2,048 for SUN-RGBD). For object points segmented from our predicted 3D mask, we randomly sample 512 points from it (if there are less than 512 points we will randomly resample to make up for the number). 
- Second, we randomly flip the frustum point cloud (after rotating the frustum to the center) along the YZ plane in camera coordinate (Z is forward, Y is pointing down). 
- Thirdly, we perturb the points by shifting the entire frustum point cloud in Z-axis direction such that the depth of points is augmented. 

Together with all data augmentation, we modify the ground truth labels for 3D mask and headings correspondingly

##### KITTI Training

##### SUN-RGBD Training

#### C. Details on RGB Detector (Sec 4.1)






---
## Code 


> [TF](https://github.com/charlesq34/frustum-pointnets)
> [pyTorch](https://github.com/LoFaiTh/frustum_pointnes_pytorch) : 미완성??

```python 
## command_prep_data.sh
python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection

## command_test_v1.sh
python train/test.py 
	--gpu 0 
	--num_point 1024 
	--model frustum_pointnets_v1 
	--model_path train/log_v1/model.ckpt 
	--output train/detection_results_v1 
	--data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle 
	--from_rgb_detection 
	--idx_path kitti/image_sets/val.txt 
	--from_rgb_detection train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_v1

python train/train.py 
	--gpu 0 
	--model frustum_pointnets_v1 
	--log_dir train/log_v1 
	--num_point 1024 
	--max_epoch 201 
	--batch_size 32 
	--decay_step 800000 
	--decay_rate 0.5

python train/train.py 
	--gpu 0 
	--model frustum_pointnets_v1 
	--log_dir train/log_v1 
	--num_point 1024 
	--max_epoch 201 
	--batch_size 32 
	--decay_step 800000 
	--decay_rate 0.5

```


---

```python 

logits, end_points = get_instance_seg_v1_net(point_cloud, one_hot_vec,is_training, bn_decay, end_points)
end_points['mask_logits'] = logits


def get_instance_seg_v1_net(point_cloud, one_hot_vec,
                            is_training, bn_decay, end_points):
    ''' 3D instance segmentation PointNet v1 network.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
        end_points: dict
    Output:
        logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
        end_points: dict
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    net = tf.expand_dims(point_cloud, 2)

    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)

    logits = tf_util.conv2d(net, 2, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    logits = tf.squeeze(logits, [2]) # BxNxC
    return logits, end_points

#-------------------------------------------------------------------------------------

# Masking
# select masked points and translate to masked points' centroid
object_point_cloud_xyz, mask_xyz_mean, end_points = point_cloud_masking(point_cloud, logits, end_points)


def point_cloud_masking(point_cloud, logits, end_points, xyz_only=True):
    ''' Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.
    
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        logits: TF tensor in shape (B,N,2)
        end_points: dict
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: TF tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean: TF tensor in shape (B,3)
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    mask = tf.slice(logits,[0,0,0],[-1,-1,1]) < \
        tf.slice(logits,[0,0,1],[-1,-1,1])
    mask = tf.to_float(mask) # BxNx1
    mask_count = tf.tile(tf.reduce_sum(mask,axis=1,keep_dims=True),
        [1,1,3]) # Bx1x3
    point_cloud_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3]) # BxNx3
    mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1,1,3])*point_cloud_xyz,
        axis=1, keep_dims=True) # Bx1x3
    mask = tf.squeeze(mask, axis=[2]) # BxN
    end_points['mask'] = mask
    mask_xyz_mean = mask_xyz_mean/tf.maximum(mask_count,1) # Bx1x3

    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - \
        tf.tile(mask_xyz_mean, [1,num_point,1])

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = tf.slice(point_cloud, [0,0,3], [-1,-1,-1])
        point_cloud_stage1 = tf.concat(\
            [point_cloud_xyz_stage1, point_cloud_features], axis=-1)
    num_channels = point_cloud_stage1.get_shape()[2].value

    object_point_cloud, _ = tf_gather_object_pc(point_cloud_stage1,
        mask, NUM_OBJECT_POINT)
    object_point_cloud.set_shape([batch_size, NUM_OBJECT_POINT, num_channels])

    return object_point_cloud, tf.squeeze(mask_xyz_mean, axis=1), end_points

#---------------------------------------------------------------------------------

# T-Net and coordinate translation
center_delta, end_points = get_center_regression_net(object_point_cloud_xyz, one_hot_vec,is_training, bn_decay, end_points)
stage1_center = center_delta + mask_xyz_mean # Bx3
end_points['stage1_center'] = stage1_center


def get_center_regression_net(object_point_cloud, one_hot_vec,
                              is_training, bn_decay, end_points):
    ''' Regression network for center delta. a.k.a. T-Net.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in 3D mask coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        predicted_center: TF tensor in shape (B,3)
    ''' 
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3-stage1', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
        padding='VALID', scope='maxpool-stage1')
    net = tf.squeeze(net, axis=[1,2])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 256, scope='fc1-stage1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, scope='fc2-stage1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    predicted_center = tf_util.fully_connected(net, 3, activation_fn=None,
        scope='fc3-stage1')
    return predicted_center, end_points

#--------------------------------------------------------------------------------

# Get object point cloud in object coordinate
object_point_cloud_xyz_new = object_point_cloud_xyz - tf.expand_dims(center_delta, 1)
#---------------------------------------------------------------------------------


# Amodel Box Estimation PointNet
output, end_points = get_3d_box_estimation_v1_net(object_point_cloud_xyz_new, one_hot_vec,is_training, bn_decay, end_points)


def get_3d_box_estimation_v1_net(object_point_cloud, one_hot_vec,
                                 is_training, bn_decay, end_points):
    ''' 3D Box Estimation PointNet v1 network.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in object coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
    ''' 
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
        padding='VALID', scope='maxpool2')
    net = tf.squeeze(net, axis=[1,2])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, scope='fc1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, scope='fc2', bn=True,
        is_training=is_training, bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net,
        3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
    return output, end_points


#----------------------------------------------------------------
# Parse output to 3D box parameters
end_points = parse_output_to_tensors(output, end_points)
end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3


#---------------------------------------------------------------------

loss = MODEL.get_loss(labels_pl, centers_pl,
    heading_class_label_pl, heading_residual_label_pl,
    size_class_label_pl, size_residual_label_pl, end_points)


def get_loss(mask_label, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label, \
             end_points, \
             corner_loss_weight=10.0, \
             box_loss_weight=1.0):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)
        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,) 
        heading_residual_label: TF tensor in shape (B,) 
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
        logits=end_points['mask_logits'], labels=mask_label))
    tf.summary.scalar('3d mask loss', mask_loss)

    # Center regression losses
    center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    tf.summary.scalar('center loss', center_loss)
    stage1_center_dist = tf.norm(center_label - \
        end_points['stage1_center'], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    # Heading loss
    heading_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points['heading_scores'], labels=heading_class_label))
    tf.summary.scalar('heading class loss', heading_class_loss)

    hcls_onehot = tf.one_hot(heading_class_label,
        depth=NUM_HEADING_BIN,
        on_value=1, off_value=0, axis=-1) # BxNUM_HEADING_BIN
    heading_residual_normalized_label = \
        heading_residual_label / (np.pi/NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum( \
        end_points['heading_residuals_normalized']*tf.to_float(hcls_onehot), axis=1) - \
        heading_residual_normalized_label, delta=1.0)
    tf.summary.scalar('heading residual normalized loss',
        heading_residual_normalized_loss)

    # Size loss
    size_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points['size_scores'], labels=size_class_label))
    tf.summary.scalar('size class loss', size_class_loss)

    scls_onehot = tf.one_hot(size_class_label,
        depth=NUM_SIZE_CLUSTER,
        on_value=1, off_value=0, axis=-1) # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = tf.tile(tf.expand_dims( \
        tf.to_float(scls_onehot), -1), [1,1,3]) # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum( \
        end_points['size_residuals_normalized']*scls_onehot_tiled, axis=[1]) # Bx3

    mean_size_arr_expand = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32),0) # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum( \
        scls_onehot_tiled * mean_size_arr_expand, axis=[1]) # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = tf.norm( \
        size_residual_label_normalized - predicted_size_residual_normalized,
        axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    tf.summary.scalar('size residual normalized loss',
        size_residual_normalized_loss)

    # Corner loss
    # We select the predicted corners corresponding to the 
    # GT heading bin and size cluster.
    corners_3d = get_box3d_corners(end_points['center'],
        end_points['heading_residuals'],
        end_points['size_residuals']) # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2), [1,1,NUM_SIZE_CLUSTER]) * \
        tf.tile(tf.expand_dims(scls_onehot,1), [1,NUM_HEADING_BIN,1]) # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum( \
        tf.to_float(tf.expand_dims(tf.expand_dims(gt_mask,-1),-1)) * corners_3d,
        axis=[1,2]) # (B,8,3)

    heading_bin_centers = tf.constant( \
        np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), dtype=tf.float32) # (NH,)
    heading_label = tf.expand_dims(heading_residual_label,1) + \
        tf.expand_dims(heading_bin_centers, 0) # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(hcls_onehot)*heading_label, 1)
    mean_sizes = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0) # (1,NS,3)
    size_label = mean_sizes + \
        tf.expand_dims(size_residual_label, 1) # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum( \
        tf.expand_dims(tf.to_float(scls_onehot),-1)*size_label, axis=[1]) # (B,3)
    corners_3d_gt = get_box3d_corners_helper( \
        center_label, heading_label, size_label) # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        center_label, heading_label+np.pi, size_label) # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1),
        tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0) 
    tf.summary.scalar('corners loss', corners_loss)

    # Weighted sum of all losses
    total_loss = mask_loss + box_loss_weight * (center_loss + \
        heading_class_loss + size_class_loss + \
        heading_residual_normalized_loss*20 + \
        size_residual_normalized_loss*20 + \
        stage1_center_loss + \
        corner_loss_weight*corners_loss)
    tf.add_to_collection('losses', total_loss)

    return total_loss

```


---

https://medium.com/@yckim/%EC%A0%95%EB%A6%AC-roarnet-a-robust-3d-object-detection-based-on-region-approximation-refinement-91c66201eaf2
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA5NzQwMjcwNl19
-->