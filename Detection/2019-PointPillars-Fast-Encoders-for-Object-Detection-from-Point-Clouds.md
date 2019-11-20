# [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

https://researchcode.com/code/3089170440/pointpillars-fast-encoders-for-object-detection-from-point-clouds/

> CVPR2019, SECOND기반 



## [영문정리](https://medium.com/@m7807031/pointpillars-fast-encoders-for-object-detection-from-point-clouds-brief-3c868c5c463d)

![](https://i.imgur.com/BzJHmN3.png)

구성 요소 3가지 `It can be seen that PP is mainly composed of three major parts:  `
1. Pillar Feature Net -> 2. Backbone (2D CNN) -> 3. Detection Head (SSD)

**1. Pillar Feature Net**  
- Pillar Feature Net will first scan all the point clouds with the **overhead view** and build the pillars per unit of xy grid. 
	- It is also the basic unit of the feature vector of PP. 
- By calculating the point cloud in each Pillar, you can get D = for each point cloud. [x, y, z, r, xc, yc, zc, xp, yp]  among them:  
	- x,y,z,r is a single cloud x, y, z, reflection  
	- Xc, yc, zc is the point cloud point from the geometric center point of the pillar  
	- Xp, yp is the distance from the center of pillar x, y

Then combine the information into [D, P, N] superimposed tensors  among them:  
- D is the point cloud D  
- P is Pillar’s index  
- N is the point cloud index of the Pillar

Next,  
- Use a simple version of pointNet (a linear layer containing Batch-Norm and ReLu) to generate (C, P, N) tensors, 
- then generate (C, P) tensors, 
- and finally produce (C, H, W) Point cloud feature image (2D).

**2. Backbone**  
Can refer to the picture for calculation

**3. Detection Head**  
PP uses Single Shot Detector for bounding-box calculations

---

## [영문정리2](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/point_pillars.html)

라이다 데이터를 **필라**형태로 묶고, 포인트넷으로 인코딩 하여 2D BEV pseudo-이미지 생성. `Group lidar data into pillars and encode them with pointnet to form a 2D birds view pseudo-image.` 2D 이미지 탐지 네트워크인 SSD를 이용하여 물체 탐지 수행 `SSD with 2D convolution is used to process this pseudo image for object detection. `It achivees the SOTA at 115 Hz.

#### Overall impression

This paper follows the line of work of [VoxelNet](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/voxelnet.md) and [SECOND](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/second.md) and improves the encoding methods. 

Both voxelnet and SECOND encode point cloud into **3D voxels** and uses expensive 3D convolution. The main contribution of this paper lies in that it encodes (“sacrifices”) the information of the relatively unimportant dimension of z into different channels of the 2D pseudo image. This greatly boosts the inference.

#### Key ideas[](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/point_pillars.html#key-ideas)

-  복셀대신 **필라**상에 동작 한다.  PointPillars operates on pillars instead of voxels and eliminates the need to tune binning of the vertical direction by hand. All operation on the pillars are 2D conv, which can be highly optimized on GPU.

-   The sparsity of point cloud bird’s eye view is exploited by fixed number of non-empty pillars per sample (P=12000) and number of points per pillar (N=100). This creates a tensor of (D, P, N). D=9 is the dimension of the augmented (“decorated”) lidar point (x, y, z, reflectance r, center of mass of all points in the pillar xc, yc, zc, offset form xc and yc, xp and yp).

-   Pillar encoding (D, P, N) –> (C, P, N) –> (C, P) –> (C, H, W). The last step is to scatter the pillars back to the 2D image.

-   Detection head uses SSD with 2D anchor boxes. Bounding box height and elevation in z direction were not used for anchor box matching but rather used as additional regression target.

-   Loss is standard object detection loss (with additional parameters h, l and heading angle $\theta$. The encoding of heading angle is $\sin\theta$ and cannot differentiate opposite directions with difference of $\pi$. Additional heading classification is used.

-   Adjusting the size of spatial binning will impact the speed (and accuracy).

-   **Point cloud encoding method**:
    -   MV3D uses M+2 channel BV images
    -   Complex YOLO uses 1+2, largely follows MV3D
    -   PIXOR pixelates into 3D voxels of occupancy (0 or 1), but treats z as diff channels of 2D input
    -   VoxelNet pixelates into 3D voxels, but the pixelation process is by PointNet
    -   PointPillars pixelates into 2D voxels, but the pixelation process is by PointNet

#### Technical details[](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/point_pillars.html#technical-details)

-   A lidar robotics pipeline uses a bottom up approach involving BG subtraction, spatiotemporal clustering and classification.

-   A common way to process point cloud is to project it to regular 2D grid from bird’s eye view.  [MV3D](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/mv3d.html)  and  [AVOD](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/avod.html)  relies on fixed feature extractors. VoxelNet is the first end-to-end method to learn features in each voxel bin of a point cloud. SECOND improves upon VoxelCloud and speeds up the inference.
-   The point cloud is a sparse representation and an image is dense. Bird’s eye view is extremely sparse, but also creates opportunities for extreme speedup.
-   Lidar point cloud not in the FOV of camera is discarded.
-   Lidars are usually mounted on top of a car and the mass center of z is ~0.6 m.
-   TensorRT gives about 50% speedup from PyTorch.
-   KITTI lidar image is 40m x 80m.
-   Adding the off-center offset xp and yp would actually boost AP. So yes,  **representation matters**.

#### Notes[](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/point_pillars.html#notes)

-   This work can be applied to radar and multiple lidar data.
-   The encoding of heading angle can be improved by using both $\sin\theta$ and $\cos\theta$, as in AVOD.
-   The  [code](https://github.com/nutonomy/second.pytorch)  is based on the code base of SECOND.

---


(https://github.com/nutonomy/second.pytorch)

Abstract - 물체 탐지는 로봇 등에 중요 기술이다. 본 논문에서는 탐지 작업을 위해 유용한 포맷으로 점군을 인코딩시의 문제점을 기술 한다. ` Object detection in point clouds is an important aspect of many robotics applications such as autonomous driving. In this paper we consider the problem of encoding a point cloud into a format appropriate for a downstream detection pipeline. `

두가지 인코더가 제안 되고 있다. `Recent literature suggests two types of encoders;` 
- fixed encoders tend to be fast but sacrifice accuracy, 
- while encoders that are learned from data are more accurate, but slower. 

> 인코더는 특징 추출기를 의미 하는가? 

본 논문에서는 PointNET을 이용하여서 기둥 모양(pillars)의 특징 표현방식을 제안 한다. `In this work we propose PointPillars, a novel encoder which utilizes PointNets to learn a representation of point clouds organized in vertical columns (pillars). `

인코딩된 특징들은 2D CNN에 적용가능하다. `While the encoded features can be used with any standard 2D convolutional detection architecture, we further propose a lean downstream network. `

성능 평가 결과 좋고 빠르다. `Extensive experimentation shows that PointPillars outperforms previous encoders with respect to both speed and accuracy by a large margin. Despite only using lidar, our full detection pipeline significantly outperforms the state of the art, even among fusion methods, with respect to both the 3D and bird’s eye view KITTI benchmarks. This detection performance is achieved while running at 62 Hz: a 2 - 4 fold runtime improvement. A faster version of our method matches the state of the art at 105 Hz. These benchmarks suggest that PointPillars is an appropriate encoding for object detection in point clouds.`

## 1. Introduction

자율 주행 차에서 탐지 기술은 중요 하다. 기존 방법은 배경제거 - 군집화 - 분류의 형식이다.  `Deploying autonomous vehicles (AVs) in urban environments poses a difficult technological challenge. Among other tasks, AVs need to detect and track moving objects such as vehicles, pedestrians, and cyclists in realtime. To achieve this, autonomous vehicles rely on several sensors out of which the lidar is arguably the most important. A lidar uses a laser scanner to measure the distance to the environment, thus generating a sparse point cloud representation. Traditionally, a lidar robotics pipeline interprets such point clouds as object detections through a bottomup pipeline involving background subtraction, followed by spatiotemporal clustering and classification [12, 9].`

딥러닝의 발전으로 라이다 점군 기반 연구가 많이 진행 되었따. `Following the tremendous advances in deep learning methods for computer vision, a large body of literature has investigated to what extent this technology could be applied towards object detection from lidar point clouds [31, 29, 30, 11, 2, 21, 15, 28, 26, 25].`

(이미지 분석 기술대비) 대부분 비슷하지만 큰 차이점은 다음과 같다. `  While there are many similarities between the modalities, there are two key differences: `
- 1) the point cloud is a sparse representation, while an image is dense and 
- 2) the point cloud is 3D, while the image is 2D. 

결과적으로 이미지에서 사용하는 CNN을 라이다에 적용하기 어렵다. `As a result object detection from point clouds does not trivially lend itself to standard image convolutional pipelines.`

초기 연구에서는 3D CNN을 이용하거나 점군을 이미지로 투영 하는 방법에 초점을 두었다. 최근 연구는 BEV방식을 사용한다. `Some early works focus on either using 3D convolutions [3] or a projection of the point cloud into the image [14]. Recent methods tend to view the lidar point cloud from a bird’s eye view [2, 11, 31, 30]. ` BEV의 장점은 아래와 같다. `This overhead perspective offers several advantages such as lack of scale ambiguity and the near lack of occlusion.`

하지만 BEV는 **extremely sparse **한 속성으로 실 적용하기에는 효율성이 낮다. `However, the bird’s eye view tends to be extremely sparse which makes direct application of convolutional neural networks impractical and inefficient. `

이에 대한 해결방법으로는 공간을 그리드로 나누고 **hand crafted feature**를 각 그리드에 적용하는 것이다. `A common workaround to this problem is to partition the ground plane into a regular grid, for example 10 x 10 cm, and then perform a hand crafted feature encoding method on the points in each grid cell [2, 11, 26, 30]. `

하지만 이 hard-coded feature방법은 일반화 성능이 약하다. `However, such methods may be sub-optimal since the hard-coded feature extraction method may not generalize to new configurations without significant engineering efforts. `

이 문제를 해결하기 위해 VoxelNet(PointNet기반)이 제안되었다. `To address these issues, and building on the PointNet design developed by Qi et al. [22], VoxelNet [31] was one of the first methods to truly do endto-end learning in this domain. `

VoxelNet은 각 복셀별로 PointNet을 적용하였다. 하지만 속도가 느리다. SEOND가 속도 문제를 개선 하였지만 3D CNN은 아직도 병목 지점으로 남아 있다. `VoxelNet divides the space into voxels, applies a PointNet to each voxel, followed by a 3D convolutional middle layer to consolidate the vertical axis, after which a 2D convolutional detection architecture is applied. While the VoxelNet performance is strong, the inference time, at 4.4 Hz, is too slow to deploy in real time. Recently SECOND [28] improved the inference speed of VoxelNet but the 3D convolutions remain a bottleneck.`

본 논문에서는 2D CNN 레이어 만으로 3D 물체 탐지가 가능한 **PointPillars**를 제안 하였다. `In this work we propose PointPillars: a method for object detection in 3D that enables end-to-end learning with only 2D convolutional layers. `

제안 방식은 pillars의 특징을 학습하여 3D 물체의 박스를 예측 할수 있다. `PointPillars uses a novel encoder that learn features on pillars (vertical columns) of the point cloud to predict 3D oriented boxes for objects. `

장점은 아래와 같다. `There are several advantages of this approach.`
- Fixed encoder가 아니라 특징을 학습 하는 방식이라 점군의 정보를 잘 반영한다. `First, by learning features instead of relying on fixed encoders, PointPillars can leverage the full information represented by the point cloud. `
- 복셀이 아니라 필라에 대하여 연산을 함으로써 세로 방향의 Binning에 대하여 조정이 불필요 하다. `Further, by operating on pillars instead of voxels there is no need to tune the binning of the vertical direction by hand. `
- 필라는 GPU에서 동작시 더 효율적이다. `Finally, pillars are highly efficient because all key operations can be formulated as 2D convolutions which are extremely efficient to compute on a GPU. `
- 학습 기반 방식의 또 다른 장점은 다른 점군환경에 대하여 수작업 조정이 불필요 하다는 것이다. `An additional benefit of learning features is that PointPillars requires no hand-tuning to use different point cloud configurations. `
	- For example, it can easily incorporate multiple lidar scans, or even radar point clouds.

성능 평가 결과도 좋다. `We evaluated our PointPillars network on the public KITTI detection challenges which require detection of cars, pedestrians, and cyclists in either the bird’s eye view (BEV) or 3D [5]. While our PointPillars network is trained using only lidar point clouds, it dominates the current state of the art including methods that use lidar and images, thus establishing new standards for performance on both BEV and 3D detection (Table 1 and Table 2). At the same time PointPillars runs at 62 Hz, which is orders of magnitude faster than previous art. PointPillars further enables a trade off between speed and accuracy; in one setting we match state of the art performance at over 100 Hz (Figure 5). We have also released code that can reproduce our results.`

### 1.1. Related Work

We start by reviewing recent work in applying convolutional neural networks toward object detection in general, and then focus on methods specific to object detection from lidar point clouds.

#### 1.1.1 Object detection using CNNs

Starting with the seminal work of Girshick et al. [6] it was established that convolutional neural network (CNN) architectures are state of the art for detection in images. 

The series of papers that followed [24, 7] advocate a **two-stage approach** to this problem, where in the first stage a region proposal network (RPN) suggests candidate proposals. Cropped and resized versions of these proposals are then classified by a second stage network. Two-stage methods dominated the important vision benchmark datasets such as COCO [17] over single-stage architectures originally proposed by Liu et al. [18]. 

In a **single-stage architecture** a dense set of anchor boxes is regressed and classified in a single stage into a set of predictions providing a fast and simple architecture. Recently Lin et al. [16] convincingly argued that with their proposed focal loss function a single stage method is superior to two-stage methods, both in terms of accuracy and runtime. 

In this work, we use a single stage method.

#### 1.1.2 Object detection in lidar point clouds

3D CNN방식은 속도가 느리다. `Object detection in point clouds is an intrinsically three dimensional problem. As such, it is natural to deploy a 3D convolutional network for detection, which is the paradigm of several early works [3, 13]. While providing a straightforward architecture, these methods are slow; e.g. Engelcke et al. [3] require 0.5s for inference on a single point cloud. `

최근 속도 향상 방식은 Ground & Image Plane 방식이다. `Most recent methods improve the runtime by projecting the 3D point cloud either onto the ground plane [11, 2] or the image plane [14].`

일반적인 방법은 복셀을 이용하여 이미지 탐지기를 이용하는 것이다. `In the most common paradigm the point cloud is organized in voxels and the set of voxels in each vertical column is encoded into a fixed-length, handcrafted, feature encoding to form a pseudo-image which can be processed by a standard image detection architecture. `

fixed encoding paradigm방식 들 : `Some notable works here include MV3D [2], AVOD [11], PIXOR [30] and Complex YOLO [26] which all use variations on the same fixed encoding paradigm as the first step of their architectures. `
- The first two methods additionally fuse the lidar features with image features to create a multimodal detector. The fusion step used in MV3D and AVOD forces them to use two-stage detection pipelines, 
- while PIXOR and Complex YOLO use single stage pipelines..

PointNET이 제안 되었지만 속도가 느리다. `In their seminal work Qi et al. [22, 23] proposed a simple architecture, PointNet, for learning from unordered point sets, which offered a path to full end-to-end learning. VoxelNet [31] is one of the first methods to deploy PointNets for object detection in lidar point clouds. In their method, PointNets are applied to voxels which are then processed by a set of 3D convolutional layers followed by a 2D backbone and a detection head. This enables end-to-end learning, but like the earlier work that relied on 3D convolutions, VoxelNet is slow, requiring 225ms inference time (4.4 Hz) for a single point cloud. `

Another recent method, **Frustum PointNet [21]**, uses PointNets to segment and classify the point cloud in a frustum generated from projecting a detection on an image into 3D. Frustum PointNet’s achieved high benchmark performance compared to other fusion methods, but its multi-stage design makes end-to-end learning impractical. 

Very recently **SECOND [28]** offered a series of improvements to VoxelNet resulting in stronger performance and a much improved speed of 20 Hz. However, they were unable to remove the expensive 3D convolutional layers.

### 1.2. Contribution

- We propose a novel point cloud encoder and network, PointPillars, that operates on the point cloud to enable end-to-end training of a 3D object detection network. 

- We show how all computations on pillars can be posed as dense 2D convolutions which enables inference at 62 Hz; a factor of 2-4 times faster than other methods. 

- We conduct experiments on the KITTI dataset and demonstrate state of the art results on cars, pedestrians, and cyclists on both BEV and 3D benchmarks. 

- We conduct several ablation studies to examine the key factors that enable a strong detection performance.


## 2. PointPillars Network

![](https://i.imgur.com/kSrInWg.png)
```
Figure 2. Network overview. 
- The main components of the network are a Pillar Feature Network, Backbone, and SSD Detection Head. See Section 2 for more details. 
- The raw point cloud is converted to a stacked pillar tensor and pillar index tensor. 
- The encoder uses the stacked pillars to learn a set of features that can be scattered back to a 2D pseudo-image for a convolutional neural network. 
- The features from the backbone are used by the detection head to predict 3D bounding boxes for objects. 
- Note: here we show the backbone dimensions for the car network.
```

제안 방식은 점군을 입력으로 받아 출력으로 3D의 방향을 추론한다. `PointPillars accepts point clouds as input and estimates oriented 3D boxes for cars, pedestrians and cyclists.`

구성 3단계 ` It consists of three main stages (Figure 2): `
- (1) A feature encoder network that converts a point cloud to a sparse pseudoimage; 
- (2) a 2D convolutional backbone to process the pseudo-image into high-level representation; 
- and (3) a detection head that detects and regresses 3D boxes.

## 2.1. Pointcloud to Pseudo-Image

2D CNN적용을 위해 점군을 pseudo-이미지로 변경 하여야 한다. `To apply a 2D convolutional architecture, we first convert the point cloud to a pseudo-image. `

We denote by l a point in a point cloud with coordinates x, y, z and reflectance r. 

As a first step the point cloud is discretized into an evenly spaced grid in the x-y plane, creating a set of pillars P with |P| = B. 

> z축에 대하여서는 파라미터가 불필요 하다. `Note that there is no need for a hyper parameter to control the binning in the z dimension. `

The points in each pillar are then augmented with xc, yc, zc, xp and yp 
- where the `c` subscript denotes **distance** to the arithmetic mean of all points in the pillar 
- and the `p` subscript denotes the **offset** from the pillar x, y center. 

The augmented lidar point l is now D = 9 dimensional.

점군의 sparsity 속성상 대부분의 필라는 비어 있거나 적은수의 점들로 구성되어 있을것이다. `The set of pillars will be mostly empty due to sparsity of the point cloud, and the non-empty pillars will in general have few points in them. `
- For example, at 0.162 m2 bins the point cloud from an HDL-64E Velodyne lidar has 6k-9k non-empty pillars in the range typically used in KITTI for ∼ 97% sparsity. 

This sparsity is exploited by imposing a limit both on the number of non-empty pillars per sample (P) and on the number of points per pillar (N) to create a dense tensor of size (D, P, N). 

If a sample or pillar holds too much data to fit in this tensor the data is randomly sampled. Conversely, if a sample or pillar has too little data to populate the tensor, zero padding is applied.

간단한 버젼의 **PointNET** 설계 `Next, we use a simplified version of PointNet where, for each point, a linear layer is applied followed by BatchNorm [10] and ReLU [19] to generate a (C, P, N) sized tensor. This is followed by a max operation over the channels to create an output tensor of size (C, P). Note that the linear layer can be formulated as a 1x1 convolution across the tensor resulting in very efficient computation.`

Once encoded, the features are scattered back to the original pillar locations to create a pseudo-image of size (C, H, W) where H and W indicate the height and width of the canvas.

### 2.2. Backbone

We use a similar backbone as [31] and the structure is shown in Figure 2. 

The backbone has two sub-networks: 
- one top-down network that produces features at increasingly small spatial resolution 
- and a second network that performs upsampling and concatenation of the top-down features. 

The top-down backbone can be characterized by a series of blocks Block(S, L, F). Each block operates at stride S (measured relative to the original input pseudo-image). A block has L 3x3 2D conv-layers with F output channels, each followed by BatchNorm and a ReLU. The first convolution inside the layer has stride S Sin to ensure the block operates on stride S after receiving an input blob of stride Sin. All subsequent convolutions in a block have stride 1.

The final features from each top-down block are combined through upsampling and concatenation as follows. 
- First, the features are upsampled, Up(Sin, Sout, F) from an initial stride Sin to a final stride Sout (both again measured wrt. original pseudo-image) using a transposed 2D convolution with F final features. 
- Next, BatchNorm and ReLU is applied to the upsampled features. 
- The final output features are a concatenation of all features that originated from different strides.

### 2.3. Detection Head 

In this paper, we use the **Single Shot Detector (SSD)** [18] setup to perform 3D object detection. 

Similar to SSD, we match the prior boxes to the ground truth using **2D Intersection over Union (IoU) [4]**. 

Bounding box height and elevation were not used for matching; instead given a 2D match, the height and elevation become additional regression targets.

## 3. Implementation Details

In this section we describe our network parameters and the loss function that we optimize for.

### 3.1. Network 

Instead of pre-training our networks, all weights were **initialized randomly** using a uniform distribution as in [8]. 

The encoder network has C = 64 output features. 

The car and pedestrian/cyclist backbones are the same except for the stride of the first block (S = 2 for car, S = 1 for pedestrian/cyclist). 

Both network consists of three blocks, 
- Block1(S, 4, C), 
- Block2(2S, 6, 2C), 
- and Block3(4S, 6, 4C). 

Each block is upsampled by the following upsampling steps: Up1(S, S, 2C), Up2(2S, S, 2C) and Up3(4S, S, 2C). 

Then the features of Up1, Up2 and Up3 are concatenated together to create 6C features for the detection head. 

### 3.2. Loss 

We use the same **loss functions** introduced in **SECOND [28]**. 

Ground truth boxes and anchors are defined by (x, y, z, w, l, h, θ). The localization regression residuals between ground truth and anchors are defined by:

> 중략 

## 4. Experimental setup 

### 4.1. Datase

### 4.2. Setting

### 4.3. Data Augmentation


Data augmentation is critical for good performance on the KITTI benchmark [28, 30, 2]. 

First, following SECOND [28], we create a lookup table of the ground truth 3D boxes for all classes and the associated point clouds that falls inside these 3D boxes. 

Then for each sample, we randomly select 15, 0, 8 ground truth samples for cars, pedestrians, and cyclists respectively and place them into the current point cloud. We found these settings to perform better than the proposed settings [28]. 

Next, all ground truth boxes are individually augmented. Each box is rotated (uniformly drawn from [−π/20, π/20]) and translated (x, y, and z independently drawn from N (0, 0.25)) to further enrich the training set. 

Finally, we perform two sets of global augmentations that are jointly applied to the point cloud and all boxes. 
- First, we apply random mirroring flip along the x axis [30], then a global rotation and scaling [31, 28]. 
- Finally, we apply a global translation with x, y, z drawn from N (0, 0.2) to simulate localization noise.


## 5. Results

## 6. Realtime Inference

## 7. Ablation Studies

In this section we provide ablation studies and discuss our design choices compared to the recent literature.

### 7.1. Spatial Resolution

A trade-off between speed and accuracy can be achieved by varying the size of the spatial binning. Smaller pillars allow finer localization and lead to more features, while larger pillars are faster due to fewer non-empty pillars (speeding up the encoder) and a smaller pseudo-image (speeding up the CNN backbone). 

To quantify this effect we performed a sweep across grid sizes. From Figure 5 it is clear that the larger bin sizes lead to faster networks; at 0.282 we achieve 105 Hz at similar performance to previous methods. 

The decrease in performance was mainly due to the pedestrian and cyclist classes, while car performance was stable across the bin sizes.


###  7.2. Per Box Data Augmentation 

Both VoxelNet [31] and SECOND [28] recommend extensive **per box** augmentation. 

However, in our experiments, minimal box augmentation worked better. 

In particular, the detection performance for pedestrians degraded significantly with more box augmentation. 

Our hypothesis is that the introduction of ground truth sampling mitigates the need for extensive per box augmentation. 

###  7.3. Point Decorations 

During the lidar point decoration step, we perform the VoxelNet [31] decorations plus two additional decorations: xp and yp which are the x and y offset from the pillar x, y center. 

These extra decorations added 0.5 mAP to final detection performance and provided more reproducible experiments. 

###  7.4. Encoding 

To assess the impact of the proposed PointPillar encoding in isolation, we implemented several encoders in the official codebase of SECOND [28]. 

For details on each encoding, we refer to the original papers. 

As shown in Table 4, learning the feature encoding is strictly superior to fixed encoders across all resolution. This is expected as most successful deep learning architectures are trained end-to-end. Further, the differences increase with larger bin sizes where the lack of expressive power of the fixed encoders are accentuated due to a larger point cloud in each pillar. Among the learned encoders VoxelNet is marginally stronger than PointPillars. However, this is not a fair comparison, since the VoxelNet encoder is orders of magnitude slower and has orders of magnitude more parameters. When the comparison is made for a similar inference time, it is clear that PointPillars offers a better operating point (Figure 5). 

There are a few curious aspects of Table 4. First, despite notes in the original papers that their encoder only works on cars, we found that the MV3D [2] and PIXOR [30] encoders can learn pedestrians and cyclists quite well. Second, our implementations beat the respective published results by a large margin (1 − 10 mAP). While this is not an apples to apples comparison since we only used the respective encoders and not the full network architectures, the performance difference is noteworthy. We see several potential reasons. For VoxelNet and SECOND we suspect the boost in performance comes from improved data augmentation hyperparameters as discussed in Section 7.2. Among the fixed encoders, roughly half the performance increase can be explained by the introduction of ground truth database sampling [28], which we found to boost the mAP by around 3% mAP. The remaining differences are likely due to a combination of multiple hyperparameters including network design (number of layers, type of layers, whether to use a feature pyramid); anchor box design (or lack thereof [30]); localization loss with respect to 3D and angle; classification loss; optimizer choices (SGD vs Adam, batch size); and more. However, a more careful study is needed to isolate each cause and effect.

## 8. Conclusion

In this paper, we introduce PointPillars, a novel deep network and encoder that can be trained end-to-end on lidar point clouds. We demonstrate that on the KITTI challenge, PointPillars dominates all existing methods by offering higher detection performance (mAP on both BEV and 3D) at a faster speed. Our results suggests that PointPillars offers the best architecture so far for 3D object detection from lidar.