| 논문명 | Multi-View 3D Object Detection Network for Autonomous Driving |
| --- | --- |
| 저자\(소속\) | Xiaozhi Chen \(칭화대\), Bo Li \(Baidu\) |
| 학회/년도 | 2017, [논문](https://arxiv.org/pdf/1611.07759.pdf) |
| 키워드 | Region-based feature fusion |
| 참고 | [홈페이지](https://xiaozhichen.github.io), [CVPR2017](https://www.youtube.com/watch?v=ChkgSvxAvMg&list=PL_bDvITUYucD6jDTC02jwuYDqIzkRsSIM&index=11), [Youtube](https://youtu.be/POqBiiLaslk), [MobileNets](https://www.gitbook.com/book/adioshun/deep_drive/edit#/edit/master/papermovilenet.md?_k=rydhgy) |
| 코드 | [저자코드](https://github.com/prclibo/kitti_eval), [MV3D](https://github.com/bostondiditeam/MV3D), [3rd Reader](https://github.com/hengck23/didi-udacity-2017) |

> !!!!! 3. MV3D Network 부터 읽기 시작 \(2017.09.21\)

# Multi-View 3D CNN for Vehicle Detection

## 0. Abstract

목표 : A sensory-fusion framework

* 입력 : LIDAR point cloud + RGB images
* 출력 : predicts oriented 3D bounding boxes

구성

* 3D object proposal generation : Generates 3D candidate boxes

  * from the bird’s eye view representation of 3D point cloud. 

* multi-view feature fusion

특징

* We design a **deep fusion** scheme to combine region-wise features from multiple views and enable interactions between intermediate layers of different paths.

목표

* We aim at highly accurate **3D localization** and **recognition of objects** in the road scene.

## 1. Introduction

### 1.1 기존 연구

각 센서들의 장점

* Laser scanners :  accurate depth information 
* cameras : detailed semantic information

최근 LIDAR기반 방식들 기술 동향

* Place 3D windows in 3D voxel grids to score the point cloud \[26, 7\] 
* Apply CNN to the front view point map in a dense box prediction scheme \[17\].

  ```
  [26] D. Z. Wang and I. Posner. Voting for voting in online point cloud object detection. In Proceedings of Robotics: Science and Systems, 2015
  [7] M. Engelcke, D. Rao, D. Zeng Wang, C. Hay Tong, and I. Posner. Vote3Deep: Fast Object Detection in 3D Point Clouds Using Efficient Convolutional Neural Networks. arXiv:1609.06666, 2016
  [17] B. Li, T. Zhang, and T. Xia. Vehicle detection from 3d lidar using fully convolutional network. In Robotics: Science and Systems, 2016
  ```

최근 Image기반 방식들 기술 동향

* Image-based methods \[4, 3\] typically first generate 3D box proposals and then perform region-based recognition using the Fast RCNN \[10\] pipeline.

  ```
  [4] X. Chen, K. Kundu, Y. Zhu, A. Berneshawi, H. Ma, S. Fidler, and R. Urtasun. 3d object proposals for accurate object class detection. In NIPS, 2015
  [3] X. Chen, K. Kundu, Z. Zhang, H. Ma, S. Fidler, and R. Urtasun. Monocular 3d object detection for autonomous driving. In CVPR, 2016
  ```

최근 Fusion\(LiDAR + Image\) 기반 방식들 기술 동향

* \[11, 8\] combine LIDAR and images for 2D detection by employing **early or late fusion schemes**. 
* \[참고\] 본 논문은 Deep fusion schemes 이용

  ```
  [11] A. Gonzalez, D. Vazquez, A. Lopez, and J. Amores. Onboard object detection: Multicue, multimodal, and multiview random forest of local experts. In IEEE Transactions on Cybernetics,
  2016
  [8] M. Enzweiler and D. M. Gavrila. A multilevel mixture-of experts framework for pedestrian classification. IEEE Transactions on Image Processing, 20(10):2967–2979, 2011
  ```

### 1.2 본 논문 제안

The main idea for utilizing multimodal information is to perform `region-based` feature fusion.

![](http://i.imgur.com/RDijuv4.png)

| MV3D | a 3D Proposal Network | + | a Region-based Fusion Network. |
| --- | --- | --- | --- |


제안 \#1 : multi-view encoding scheme

* to obtain a compact and effective representation for sparse 3D point cloud.

#### A. 3D proposal network

* 목적 : 포인트 클라우드의 bird’s eye view를 이용하여서 3D 후보 영역을 추천 한다. `The 3D proposal network utilizes a bird’s eye view representation of point cloud to generate highly accurate 3D candidate boxes.`

* 장점 : The benefit of \[3D object proposals\] is that it can be projected to **any views** in 3D space.

#### B. Region-based Fusion Network

* 목적 : 3D 후보영역을 mulitple views에서 Feature map으로 투영하여 region-wise feature들을 추출 한다. 'The multi-view fusion network extracts region-wise features by projecting 3D proposals to the feature maps from mulitple views.'

* 특징 : We design a `deep fusion` approach to enable interactions of intermediate layers from different views.

###### early/late fusion scheme 대비 deep fusion 성능 향상을 위해 사용한 방법

* Drop-path training \[15\] 
* Auxiliary loss

```
[15] G. Larsson, M. Maire, and G. Shakhnarovich. Fractalnet:Ultra-deep neural networks without residuals.arXiv:1605.07648, 2016.
```

## 2. 관련 연구

### 2.1 3D Object Detection in Point Cloud

대부분의 방법들은 3D 포인트 클라우드를 Voxel 그리드로 인코딩 시키는 것이다. `Most existing methods encode 3D point cloud with voxel grid representation.`

* Sliding Shapes \[22\] and Vote3D \[26\] apply SVM classifers on 3D grids encoded with geometry features.
* Some recently proposed methods \[23, 7, 16\] improve feature representation with 3D convolutions.

* In addition to the 3D voxel representation, VeloFCN \[17\] projects point cloud to the front view, obtaining a 2D point map.

  * They apply a fully convolutional network on the 2D point map and predict 3D boxes densely from the convolutional feature maps.

* \[24, 18, 12\] investigate volumetric and multi-view representation of point cloud for 3D object classification.

In this work, we encode 3D point cloud with multi-view feature maps, enabling region-based representation for multimodal fusion

```
[22] S. Song and J. Xiao. Sliding shapes for 3d object detection in depth images. In ECCV. 2014.
[26] D. Z. Wang and I. Posner. Voting for voting in online point cloud object detection. In Proceedings of Robotics: Science and Systems, 2015.
[23] S. Song and J. Xiao. Deep sliding shapes for amodal 3d object detection in rgb-d images. In CVPR, 2016. 
[7] M. Engelcke, D. Rao, D. Zeng Wang, C. Hay Tong, and I. Posner. Vote3Deep: Fast Object Detection in 3D Point Clouds Using Efficient Convolutional Neural Networks. arXiv:1609.06666, 2016.
[16] B. Li. 3d fully convolutional network for vehicle detection in point cloud. IROS, 2017. 
[17] B. Li, T. Zhang, and T. Xia. Vehicle detection from 3d lidar using fully convolutional network. In Robotics: Science and Systems, 2016.
[24] H. Su, S.Maji, E.Kalogerakis, and E. Learned-Miller. Multiview convolutional neural networks for 3d shape recognition. In ICCV, 2015. 
[18] C. R. Qi, M. N. H. Su, A. Dai, M. Yan, and L.Guibas. Volumetric and multi-view cnns for object classification on 3d data. In CVPR, 2016. 
[12] V. Hegde and R. Zadeh. Fusionnet: 3d object classification using multiple data representations. CoRR, abs/1607.05695, 2016.
```

### 2.2 3D Object Detection in Images.

* 3DVP \[28\] introduces 3D voxel patterns and employ a set of ACF detectors to do 2D detection and 3D pose estimation.

* 3DOP \[4\] reconstructs depth from stereo images and uses an energy minimization approach to generate 3D box proposals, which are fed to an R-CNN \[10\] pipeline for object recognition.

* Mono3D \[3\] shares the same pipeline with 3DOP, it generates 3D proposals from monocular images.

* \[31, 32\] introduces a detailed geometry representation of objects using 3D wire frame models.

* To incorporate temporal information, some work\[6, 21\] combine structure from motion and ground estimation to lift 2D detection boxes to 3D bounding boxes.

* Image-based methods usually rely on accurate depth estimation or landmark detection.

본 논문에서는 : Our work shows how to incorporate LIDAR point cloud to improve 3D localization.

```
[28] Y. Xiang, W. Choi, Y. Lin, and S. Savarese. Data-driven 3d voxel patterns for object category recognition. In CVPR, 2015
[4] X. Chen, K. Kundu, Y. Zhu, A. Berneshawi, H. Ma, S. Fidler, and R. Urtasun. 3d object proposals for accurate object class detection. In NIPS, 2015
[3] X. Chen, K. Kundu, Z. Zhang, H. Ma, S. Fidler, and R. Urtasun. Monocular 3d object detection for autonomous driving. In CVPR, 2016
[31] M. Z. Zia, M. Stark, B. Schiele, and K. Schindler. Detailed 3d representations for object recognition and modeling. PAMI, 2013.
[32] M. Z. Zia, M. Stark, and K. Schindler. Are cars just 3d boxes? jointly estimating the 3d shape of multiple objects. In CVPR, pages 3678–3685, 2014.
[6] V. Dhiman, Q. H. Tran, J. J. Corso, and M. Chandraker. A continuous occlusion model for road scene understanding. In CVPR, pages 4331–4339, 2016.
[21] S. Song and M. Chandraker. Joint sfm and detection cues for monocular 3d localization in road scenes. In Computer Vision and Pattern Recognition, pages 3734–3742, 2015
```

### 2.3 Multimodal Fusion

자율 주행 분야에서는 멀티모달 퓨전에 대하여서는 많은 연구가 이루어 지지는 않았다.

* \[11\] combines images, depth and optical flow using a mixture-of-experts framework for 2D pedestrian detection.

* \[8\] fuses RGB and depth images in the early stageand trains pose-based classifiers for 2D detection.

본 논문에서는 FractalNet \[15\] and Deeply-Fused Net \[27\]에 영감을 받았다.

* In FractalNet\[15\] , a base module is iteratively repeated to construct a networkwith exponentially increasing paths. 
* Similarly, \[27\] constructs deeply-fused networks by combining shallow and deep sub networks. 

위 방식과 다른점 :  `Our network differs from them by using the same base network for each column and adding auxiliary paths and losses for regularization.`

```
[11] A. Gonzalez, D. Vazquez, A. Lopez, and J. Amores. Onboard object detection: Multicue, multimodal, and multiview random forest of local experts. In IEEE Transactions on Cybernetics,
2016.
[8] M. Enzweiler and D. M. Gavrila. A multilevel mixture-of experts framework for pedestrian classification. IEEE Transactions on Image Processing, 20(10):2967–2979, 2011
[15] G. Larsson, M. Maire, and G. Shakhnarovich. Fractalnet: Ultra-deep neural networks without residuals. arXiv:1605.07648, 2016.
[27] J. Wang, Z. Wei, T. Zhang, and W. Zeng. Deeply-fused nets. arXiv:1605.07716, 2016.
```

### 2.4 3D Object Proposals

2D와 유사한 방법으로 3D에서는 Object Proposals 을 수행 한다.

* generate a smallset of 3D candidate boxes in order to cover most of the objects in 3D space.

* 3DOP \[4\] designs some depth features in stereo point cloud to score a large set of 3D candidate boxes.

* Mono3D \[3\] exploits the ground plane priorand utilizes some segmentation features to generate 3D proposals from a single image.

Both 3DOP and Mono3D \(use hand-crated\) features.

* Deep Sliding Shapes \[23\] exploits more powerful deep learning features. 
  * However, it operates on 3D voxel grids and uses computationally expensive 3D convolutions. 

본 논문에서는 : We propose a more efficient approach by introducing a bird’s eye view representation of point cloud and employing 2D convolutions to generate accurate 3D proposals.

| Dim | Paper | Method | Weekpoint |
| --- | --- | --- | --- |
| 2D | Selective search\[25\],\[33\],\[2\] |  |  |
| 3D | 3DOP \[4\] | depth features\(From stereo point cloud\) | hand-crated features |
| 3D | Mono3D \[3\] | some segmentation features\(From Image\) | hand-crated features |
| 3D | Deep Sliding Shapes \[23\] | more powerful deep learning features\(??\) | computationally expensive |
| 3D | 제안\(MV3D\) | 2D convolutions \(From bird’s eye view \) |  |

## 3. MV3D Network

###### Step 1. 입력 =  multi-view representation of 3D point cloud\(Bird Eye view, Front view\) + 이미지

###### Step 2. 3D 후보영역 선발 : bird’s eye view이용

###### Step 3. region-based representation을 통해 Multi-view representation 통합\(Fuse\)

###### Step 4. 분류 및 BBox 찾기 \(category classification and oriented 3D box regression\)

### 3.1. 3D Point Cloud Representation

기존 방식

* Usually encodes 3D LIDAR point cloud into a 3D grid \[26, 7\] 
  * Raw 데이터를 잘 유지하고 있지만, feature extraction 계산 부하가 크다. 
* Usually encodes 3D LIDAR point cloud into a front view map \[17\]. 

제안 방식

* Encodes 3D LIDAR point cloud into the bird’s eye view
* Encodes 3D LIDAR point cloud into the front view

![](http://i.imgur.com/atUf0Of.png)

#### A. Bird’s Eye View Representation.

The bird’s eye view representation is encoded by height, intensity and density.

We discretize the projected point cloud into a 2D grid with resolution of 0.1m.

##### 가. height feature

* For each cell, the height feature is computed as the maximum height of the points in the cell.

* To encode more detailed height information, the point cloud is devided equally into M slices.

* A height map is computed for each slice, thus we obtain M height maps.

##### 나. intensity feature\(빛의 반사\)

* The intensity feature is the reflectance value of the point which has the maximum height in each cell. 

##### 다. density feature

* The point cloud density indicates the number of points in each cell. 

##### 라. 3개의 Feature 처리 방법

* To normalize the feature, it is computed as $$min(1.0, \frac{log(N+1)}{log(54)})$$

  * $$N$$ = the number of points in the cell. 

* Note that the **intensity** and **density** features are computed for the whole point cloud while the **height** feature is computed for M slices, thus in total the bird’s eye view map is encoded as \(M +2\)-channel features.

#### B. Front View Representation.

* Front view는 bird’s eye view에 추가적 정보\(complementary information\)들 제공\(provide\)

* LiDAR 포인트 클라우드는 sparse하기 때문에 이미지에 투영하게 되면 sparse 2D pointmap이 생성된다.

  * `As LIDAR point cloud is very sparse, projecting it into the image plane results in a sparse 2D pointmap.`

* 따라서, 제안 방식은 `cylinder plane`에 투영하여 `dense front view map`을 생성한다. \[17:Bo Li\]

As LIDAR point cloud is very sparse, projecting it into the image plane results in a sparse 2D pointmap.

* `Instead, we project it to a cylinder plane to generate a dense front view map as in [17].`

Given a 3D point $$p = (x, y, z)$$, its coordinates $$p_{fv} = (r, c)$$ in the front view map can be computed using  
![](http://i.imgur.com/fJhY03j.png)

* ∆θ and ∆φ are the horizontal and vertical resolution of laser beams, respectively. 

We encode the front view map with three-channel features, which are **height**, **distance** and **intensity**

### 3.2 3D Proposal Network

> Faster R-CNN의 RPN에 영감을 얻음

입력 : bird’s eye view map

In 3D object detection에서 `bird’s eye view map`이 `front view/image plane`대비 가지는 장점 3가지

1. 물리적 크기 정보를 가지고 있음 : objects preserve physical sizes when projected to the bird’s eye view, thus having small size variance, which is not the case in the front view/image plane.

2. 가림 현상 제거 : objects in the bird’s eye view occupy different space, thus avoiding the occlusion problem.

3. 차량들이 바닥에 위치 하기 떄문에 위에서 내려다 보는것이 더 유용하다. in the road scene, since objects typically lie on the ground plane and have small variance invertical location, the bird’s eye view location is more crucial to obtaining accurate 3D bounding boxes.

Given a bird’s eye view map. the network generates 3D box proposals from a set of 3D prior boxes.

* Each 3D box is parameterized by $$(x, y, z, l, w, h)$$

  * \(x, y\) is the varying positions in the bird’s eye view feature map
  * z can be computed based on the camera height and object height. 

* For each 3D prior box, the corresponding bird’s eye view anchor $$(x_{bv}, y_{bv}, l_{bv}, w_{bv})$$ can be obtained by discretizing $$(x, y, l, w)$$.

  * 차량 탐지의 `prior boxes`의 \(l, w\)= {\(3.9, 1.6\), \(1.0, 0.6\)}, h = 1.56m.

* We design $$N$$ 3D prior boxes by clustering ground truth object sizes in the training set.

* By rotating the bird’s eye view anchors 90 degrees, we obtain N = 4 prior boxes.

* We do not do orientation regression in proposal generation, whereas we left it to the next prediction stage.

* The orientations of 3D boxes are restricted to {0◦, 90◦}, which are close to the actual orientations of most road scene objects.

* This simplificatio nmakes training of proposal regression easier

#### A. Upsampling / Deconvolutional

문제점 With a disretization resolution of 0.1m으로 인해서 , object boxes in the bird’s eye view only occupy 5~40 pixels.

* Detecting such extra-small objects is still a difficult problem for deep networks.

해결책 \#1 : 고해상도 이미지를 입력으로 이용 -&gt; 컴퓨팅 파워 필요

해결책 \#2 : We opt for feature map upsampling as in \[1\].

* We use 2x bilinear upsampling after the last convolution layer in the proposal network. 

In our implementation, the front-end convolutions only proceed three pooling operations, i.e., 8x downsampling.

Therefore, combined with the 2x deconvolution, the feature map fed to the proposal network is 4x downsampled with respect to the bird’s eye view input.

#### B. 3D box regression

> RMP과 같은 방식 사용

We do 3D box regression by regressing to $$t = (\Delta x, \Delta y, \Delta z, \Delta l, \Delta w, \Delta h)$$, similarly to RPN \[19\].

* $$(\Delta x, \Delta y, \Delta z)$$ are the center offsets normalized byanchor sizes
* $$(\Delta l, \Delta w, \Delta h)$$ are computed as$$∆s = log\frac{S_{GT}}{S_{Anchor}},s \in \{l,w,h\}  $$ .

* Loss계산 : we use a multi-task loss to simultaneously classify object/background and do 3D box regression.

  * In particular, we use class-entropy for the “objectness” loss and Smooth $$l_1$$\[10\] for the 3D box regression loss.

  * Background anchors are ignored when computing the box regression loss.

During training, we compute the IoU overlap between anchors and ground truth bird’s eye view boxes.

An anchor is considered to be positive if its overlap is above 0.7, and negative if the overlap is below 0.5.

Anchors with overlap in between are ignored

Lidar의 산재한 포인트 클라우드로 인해서 빈 anchor가 발생한다. 본 논문에서는 연산부하를 줄이기 위해서 학습/테스트시 빈 Anchor는 제거 하였다.

* This can be achieved by computing an integral image over the point occupancy map.
* For each non-empty anchor at each position of the last convolution feature map, the network generates a 3D box. 
* To reduce redundancy, we apply Non-Maximum Suppression \(NMS\) on the bird’s eye view boxes. 

Different from \[23\], we did not use 3D NMS because objects shouldoccupy different space on the ground plane.

We use IoU threshold of 0.7 for NMS.

The top 2000 boxes are kept during training, while in testing, we only use 300 boxes.

### 3.3. Region-based Fusion Network

역할

* **combine features** from multiple views 
* jointly **classify** object proposals and do oriented 3D **box regression**

#### A. Multi-View ROI Pooling.

목적 : 여러 view에서 오는 데이터들을 same length로 맞추기  
Since features from different views/modalities usually have different resolutions, we employ ROI pooling \[10\] for each view to obtain feature vectors of the same length.

Given the generated 3D proposals,we can project them to _any views_ in the 3D space.

* Any views =  bird’s eye view\(BV\), front view \(FV\), and the image plane \(RGB\). 

Given a 3D proposal $$p_{3D}$$, we obtain ROIs on each view via:


$$
ROI_v = T_{3D \rightarrow v}(p3d), v \in \{BV, FV, RGB\}
$$


* $$T_{3D \rightarrow v}$$ : the tranformation functions from the LIDAR coordinate system to the BV,FV,RGB

Given an input feature map x from the front-end network of each view, we obtain fixed-length features $$f_v$$ via ROI pooling:


$$
f_v = R(x, ROI_v), v \in\{BV, FV, RGB\}
$$


#### B. Deep Fusion.

![](http://i.imgur.com/1XqvE4Q.png)

서로 다른 Feature의 정보를 합치는 방법

* 기존 : usually use early fusion \[1\] or late fusion \[23, 13\]. 
* 제안 : Inspired by \[15, 27\], we employ a deep fusion approach, which fuses multi-view features hierarchically. 

##### 가. early fusion

For a network that has L layers, early fusion combines features $$\{f_v\}$$ from multiple views in the input stage:  
![](http://i.imgur.com/6BRrT0N.png)

* $$\{H_l, l = 1, ..., L\} $$ : feature transformation functions 
* $$\oplus$$ : a join operation \(e.g., concatenation, summation\)

##### 나. late fusion

In contrast, late fusion uses seperate subnetworks to learn feature transformation independently and combines their outputs in the prediction stage:

![](http://i.imgur.com/YL74QSz.png)

##### 다. deep fusion

To enable more interactions among features of the intermediate layers from different views, we design the following deep fusion process:

![](http://i.imgur.com/CWqcudp.png)

We use element-wise mean for the join operation for deep fusion since it is more flexible when combined with droppath training \[15\].

#### C. Oriented 3D Box Regression

##### 가.

* Given the fusion features of the multi-view network, we regress to oriented 3D boxes from 3D proposals.

* In particular, the regression targets are the 8 corners of 3D boxes:

  * $$t =(\Delta_{x0}, ...,\Delta_{x7},\Delta_{y0},...,\Delta_{y7},\Delta_{z0}, ...,\Delta_{z7})$$

They are encoded as the corner offsets normalized by the diagonal length of the proposal box.

24D 벡터가 불필요 해 보이지만, 연구 결과 효과 적이었다.

* `Despite such a 24-D vector representation is redundant in representing an oriented 3D box, we found that this encoding approach works better than the centers and sizes encoding approach.`

Note that our 3D box regression differs from \[23\] which regresses to axis-aligned 3D boxes.

In our model, the object orientations can be computed from the predicted 3D box corners.

##### 나. multitask loss

We use a multitask loss to jointly predict object categories and oriented 3D boxes.

As in the proposal network, the category loss uses cross-entropy and the 3D box loss uses smooth $$l_1$$.

###### During training,

* the positive/negative ROIs are determined based on the IoU overlap of brid’s eye view boxes.

* A 3D proposal is considered to be positive if the bird’s eye view IoU overlap is above 0.5, and negative otherwise.

###### During inference,

* we apply NMS on the 3D boxes after 3D bounding box regression.

* We project the 3D boxes to the bird’s eye view to compute their IoU overlap.

* We use IoU threshold of 0.05 to remove redundant boxes, which ensures objects can not occupy the same space in bird’s eye view.

#### D. Network Regularization

region-based fusion network에 사용된 두가지 Regularization 기법들

* drop-pathtraining \[15\] 
* auxiliary losses. 

##### 가. drop-pathtraining

For each iteration, we randomly choose to do `global drop-path` or `local drop-path` with a probability of 50%.

* If `global drop-path` is chosen, wes elect a single view from the three views with equal probability. 
* If `local drop-path` is chosen, paths input to each join node are randomly dropped with 50% probability. 

We ensure that for each join node at least one input path is kept.

##### 나. auxiliary losses

To further strengthen the representation capability of each view, we add auxiliary paths and losses to the network.

![](http://i.imgur.com/qs7jp5l.png)  
As shown in Fig. 4, the auxiliary paths have the same number of layers with the main network.

Each layer in the auxiliary paths shares weights with the corresponding layer in the main network.

We use the same multi-task loss, i.e. classification loss plus 3D box regression loss, to back-propagate  
each auxiliary path.

We weight all the losses including auxiliary losses equally.

중요 : 예측시에는 `auxiliary paths`사용 안함 : The auxiliary paths are removed during inference.

### 3.4. Implementation

#### A. Network Architecture.

In our multi-view network, each view has the same architecture.

기본은 VGG-16이고, 몇가지 부분을 수정 하였다.

* Channels are reduced to half of the original network.
* To handle extra-small objects, we use feature approximation to obtain high-resolution feature map. 
  * In particular, we insert a 2x bilinear upsampling layer before feeding the last convolution feature map to the 3DProposal Network. 
  * Similarly, we insert a 4x/4x/2x upsampling layer before the ROI pooling layer for theBV/FV/RGB branch.
* We remove the 4th pooling operation in the originalVGG network, thus the convolution parts of our network proceed 8x downsampling.
* In the muti-view fusion network, we add an extra fully connected layer fc8 in addition to the original fc6 and fc7 layer.

초기 파라미터는 VGG-16 network pretrained on ImageNet으로 세팅 하였다.

Despite our network has three branches, the number of parameters is about 75% of the VGG-16 network.

예측\(inference \)시 소요 시간 : 이미지당 0.36S \(GeForce Titan X GPU\)

#### 나. Input Representation.

In the case of KITTI, which provides only annotations for objects in the front view \(around 90◦ field of view\), we use point cloud in the range of \[0,70.4\] × \[-40, 40\] meters.

We also remove points that are out of the image boundaries when projected to the image plane.

For bird’s eye view, the discretization resolution is set to 0.1m, therefore the bird’s eye view input has sizeof 704×800.

Since KITTI uses a 64-beam Velodyne lasers canner, we can obtain a 64×512 map for the front viewpoints.

The RGB image is up-scaled so that the shortest size is 500.

#### 다. Training.

* The network is trained in an end-to-end fashion.

* For each mini-batch we use 1 image and sample 128 ROIs, roughly keeping 25% of the ROIs as positive.

* We train the network using SGD with a learning rate of 0.001 for 100K iterations.

* Then we reduce the learning rate to 0.0001 and train another 20K iterations.


---
LIDAR point cloud + RGB images
Multi-View 3D Object Detection Network for Autonomous Driving
- late fusion Schemes : [8] M. Enzweiler and D. M. Gavrila. A multilevel mixture-of experts framework for pedestrian classification. IEEE Transactions on Image Processing, 20(10):2967–2979, 2011- Early Fusion Scheme .: [11] A. Gonzalez, D. Vazquez, A. Lopez, and J. Amores. Onboard object detection: Multicue, multimodal, and multiview random forest of local experts. In IEEE Transactions on Cybernetics,2016


Chen et al. (2016c) combine LiDAR laser range data with RGB images for object detection.
In their approach, the sparse point cloud is encoded using a compact multi-view representation and a proposal generation network utilizes the bird’s eye view representation of the point cloud to generate 3D candidates. +

Finally, they combine region-wise features from multiple views with a deep fusion scheme




---

![](http://i.imgur.com/Myw0TVr.png)

# boston didi team

* [bostondiditeam](https://github.com/bostondiditeam/MV3D)

  * [nepal](https://github.com/s-nepal/MV3D)

  * [zxf8665905](https://github.com/zxf8665905/didi-udacity-compatition) : 추천

    * [lihua213\#1](https://github.com/lihua213/didi-udacity-compatition)

    * [lihua213\#2](https://github.com/lihua213/MV3D): Old version

* [leeyevi](https://github.com/leeyevi/MV3D_TF)

  * [Super-Tree](https://github.com/Super-Tree/MV3D_TF)

* [jinbeibei\(??\)](https://github.com/jinbeibei/mv3d_ros_interface)

## 0. 환경 준비

* install tensorflow-gpu and CUDA.

* A Nvidia GPU card with computation capability

* ubuntu \(\* Cuda7.5에 맞는 버젼은 14.04임\)
* CUDA   
    \(\*environment\_gpu.yml상 버젼 =7.5\)

  * [http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local\_installers/cuda-repo-ubuntu1404-7-5-local\_7.5-18\_amd64.deb](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb)
  * `apt-get install cuda-7.5-15`\)

* cuDNN

  * Download cuDNN v5.1  for CUDA 7.5 : [Runtime lib.](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/7.5/libcudnn5_5.1.10-1+cuda7.5_amd64-deb), [소스설치방법](https://github.com/adioshun/System_Setup/wiki/4_CUDA_CuDNN-Setup#driver--cuda-install-script)

* Python3.5 for MV3D related code

* Tensorflow-GPU\(version&gt;1.0\)  
* Python2.7 for ROS related script

```bash
# https://github.com/adioshun/gitBook_DeepDrive/blob/master/papermultiview-3d-cnn/environment_gpu.yml

# conda env create -f environment_gpu.yml --name mv3d_p3_gpu

conda create -n python35 python=3.5
conda install tensorflow-gpu opencv3 shapely scikit-learn keras Cython matplotlib simplejson numba
pip install easydict
```

### 0.1 GPU용으로 설정 변경

`src/net/lib/setup.py` and `src/lib/make.sh` : "arch=sm\_37" \#Google Cloud GPU Tesla K80

```
# Which CUDA capabilities do we want to pre-build for?
# https://developer.nvidia.com/cuda-gpus
#   Compute/shader model   Cards
#   6.1              P4, P40, Titan X so CUDA_MODEL = 61
#   6.0                    P100 so CUDA_MODEL = 60
#   5.2                    M40
#   3.7                    K80
#   3.5                    K40, K20
#   3.0                    K10, Grid K520 (AWS G2)
#   Other Nvidia shader models should work, but they will require extra startup
#   time as the code is pre-optimized for them.
CUDA_MODELS=30 35 37 52 60 61
```

test

```python
import tensorflow as tf
sess = tf.Session()
print(tf.__version__) # version more than v1.
```

## 1. 데이터 다운로드

![](http://i.imgur.com/TqGRi0G.png)

[The KITTI Vision Benchmark Suite Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)

## 1.9 수정 필요

### A.

![](http://i.imgur.com/va7Lg8J.png)

* `data/raw/kitti/` 경로 밑에 데이터 위치 
* `tracklet_labels.xml`파일은 `2011_09_26_drive_0001_sync` 하위 폴더에 위치 

### B.

`src/kitti_data/pykitti/tracklet.py L289`에서 다운 받은 파일명으로 변경

```
DEFAULT_DRIVE = '2011_09_26_drive_0001'
```

## 2. ./src/make.sh

### 2.1 실행 방법

```
cd src
source activate didi
sudo chmod 755 ./make.sh
./make.sh
```

conda create -n python27 python=2.7

> 아래 \[2.2\]를 직접 실행 하는것 추천

### 2.2 실행시 진행 내용

    #- `python ./net/lib/setup.py build_ext --inplace` : Fast R-CNN (MS)

    #- 'bash ./net/lib/make.sh` : building psroi_pooling layer

    #- build required .so files
    ln -s ./net/lib/roi_pooling_layer/roi_pooling.so ./net/roipooling_op/roi_pooling.so
    ln -s ./net/lib/nms/gpu_nms.cpython-35m-x86_64-linux-gnu.so ./net/processing/gpu_nms.cpython-35m-x86_64-linux-gnu.so
    ln -s ./net/lib/nms/cpu_nms.cpython-35m-x86_64-linux-gnu.so ./net/processing/cpu_nms.cpython-35m-x86_64-linux-gnu.so
    ln -s ./net/lib/utils/cython_bbox.cpython-35m-x86_64-linux-gnu.so ./net/processing/cython_bbox.cpython-35m-x86_64-linux-gnu.so

###### \[에러\]  nvcc 못 찾을경우

* 절대 경로로 수정 후 실행 

###### \[에러\] `arning: calling a constexpr __host__ function from a __host__ __device__ function is not allowed.`

* `make.sh`파일에 아래 flag `--expt-relaxed-constexpr` 추가 

```
if [ -d "$CUDA_PATH" ]; then
    nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
        -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
        -arch=sm_37 --expt-relaxed-constexpr
```

## 3. Preprocess data \(`./src/data.py`\)

> kitti기준, didi data 이용시 `utils/bag_to_kitti` 실행 필요

* MV3D net 학습시 필요한 입력 데이터 생성 
  * Lidar bird eye view features
  * Lidar front view features
  * RGB image
  * Ground Truth label
  * Ground bounding box coordinate
  * time stamp

```
./data/preprocessing/kitti/
    - gt_boxes3d :npy
    - gt_box_plot : png
    - gt_labels : npy
    - rgb : png
    - top : .npy.npz
    - top_image : png
```

| ![](http://i.imgur.com/bb67R50.png) | ![](http://i.imgur.com/AbdY7YU.png) |
| --- | --- |


### 3.9 수정 필요

#### A. data.py 수정

작업 환경

```bash

```

```python
#data.py

if config.cfg.USE_CLIDAR_TO_TOP:
    SharedLib = ctypes.cdll.LoadLibrary('/workspace/mv3d/src/lidar_data_preprocess/'
                                        'Python_to_C_Interface/ver3/LidarTopPreprocess.so')
#if config.cfg.USE_CLIDAR_TO_TOP:
#    SharedLib = ctypes.cdll.LoadLibrary('/home/stu/MV3D/src/lidar_data_preprocess/'
#                                        'Python_to_C_Interface/ver3/LidarTopPreprocess.so')
```

#### B. NameError: name 'MATRIX\_Mt' is not defined

```
# /MV3D/src/net/processing/boxes3d.py 상단에 추가 
# ./src/config.py L126 참고 
#rgb camera
MATRIX_Mt = ([[ 2.34773698e-04, 1.04494074e-02, 9.99945389e-01, 0.00000000e+00],
[ -9.99944155e-01, 1.05653536e-02, 1.24365378e-04, 0.00000000e+00],
[ -1.05634778e-02, -9.99889574e-01, 1.04513030e-02, 0.00000000e+00],
[ 5.93721868e-02, -7.51087914e-02, -2.72132796e-01, 1.00000000e+00]])

MATRIX_Kt = ([[ 721.5377, 0. , 0. ],
[ 0. , 721.5377, 0. ],
[ 609.5593, 172.854 , 1. ]])
```

### A. ./src/config.py

```
#if __C.DATA_SETS_TYPE=='test':
#    __C.DATA_SETS_DIR = osp.abspath('/home/stu/round12_data_test')

if __C.DATA_SETS_TYPE=='test':
    __C.DATA_SETS_DIR = osp.abspath('/workspace/mv3d')
```

### B. roi\_pooling.so을 심볼릭이 아닌 파일로 대체

> 이후에도 같은 문제가 발생 하므로 \[C\]방법 추천

```
cd ./src/net/roipooling_op
mv roi_pooling.so roi_pooling.so~
cp ../../net/lib/roi_pooling_layer/roi_pooling.so ./
```

### C. roi\_pooling.so 수정 버젼 다운 로드

1. [다운로드 roi\_pooling.so](https://github.com/CharlesShang/TFFRCNN/tree/roi_pooling/lib/roi_pooling_layer)

2. chmod +x roi\_pooling.so

## 4. train.py

###### \[에러\] `"tensorflow.python.framework.errors_impl.NotFoundError: YOUR_FOLDER/roi_pooling.so: undefined symbol: ZN10tensorflow7strings6StrCatB5cxx11ERKNS0_8AlphaNumES3"`

* it is related to compilation of roi\_pooling layer.
* A simple fix will be changing "GLIBCXX\_USE\_CXX11\_ABI=1" to "GLIBCXX\_USE\_CXX11\_ABI=0" in "src/net/lib/make.sh" \(line 17\)

OR Download and replace the .so with following file :[\[Download\]](https://github.com/smallcorgi/Faster-RCNN_TF/blob/6e2a941ac250da668cf93899dbd870cc4d838773/lib/roi_pooling_layer/roi_pooling.so), CUDA 8.0, Python 3.5

###### \[에러\] NameError: name 'data\_splitter' is not defined

> user this version of [train.py](https://raw.githubusercontent.com/lihua213/didi-udacity-compatition/development/src/train.py): for python2

###### \[에러\] "module 'tensorflow.python.ops.nn' has no attribute 'convolution'"

> `conda list | grep tensorflow`후 tensorflow\(cpu & gpu\) 버젼을 1.0 이상으로 변경

---

File Structure

```
├── data   <-- all data is stored here. (Introduced in detail below)
│   ├── predicted  <-- after prediction, results will be saved here.
│   ├── preprocessed   <-- MV3D net will take inputs from here(after data.py) 
│   └── raw <-- raw data
├── environment_cpu.yml  <-- install cpu version.
├── README.md
├── saved_model                 <--- model and weights saved here. 
├── src        <-- MV3D net related source code 
│   ├── config.py
│   ├── data.py
│   ├── didi_data
│   ├── kitti_data
│   ├── lidar_data_preprocess
│   ├── make.sh
│   ├── model.py
│   ├── mv3d_net.py
│   ├── net
│   ├── play_demo.ipynb
│   ├── __pycache__
│   ├── tracking.py   <--- prediction after training. 
│   ├── tracklets
│   └── train.py    <--- training the whole network. 
│── utils    <-- all related tools put here, like ros bag data into kitti format
│    └── bag_to_kitti  <--- Take lidar value from ROS bag and save it as bin files.
└── external_models    <-- use as a submodule, basically code from other repos.
    └── didi-competition  <--- Code from Udacity's challenge repo with slightly modification, sync with Udacity's new
     updates regularly.
```

Related data are organized in this way. \(Under /data directory\)

```
├── predicted <-- after prediction, results will be saved here.
│   ├── didi <-- when didi dataset is used, the results will be put here
│   └── kitti <-- When kitti dataset used for prediction, put the results here
│       ├── iou_per_obj.csv   <-- What will be evaluated for this competition, IoU score
│       ├── pr_per_iou.csv   <--precision and recall rate per iou, currently not be evaluated by didi's rule
│       └── tracklet_labels_pred.xml  <-- Tracklet generated from prediction pipeline. 
├── preprocessed  <-- Data will be fed into MV3D net (After processed by data.py)
│   ├── didi <-- When didi dataset is processed, save it here
│   └── kitti <-- When Kitti dataset is processed, save it here
│       ├── gt_boxes3d
│           └── 2011_09_26
│               └── 0005
|                   |___ 00000.npy
├       |── gt_labels
│           └── 2011_09_26
│               └── 0005 
|                   |___ 00000.npy
|       ├── rgb
│           └── 2011_09_26
│               └── 0005 
|                   |___ 00000.png
|       ├── top
│           └── 2011_09_26
│               └── 0005 
|                   |___ 00000.npy
|       └── top_image
|           └── 2011_09_26
|               └── 0005 
|                   |___ 00000.png
└── raw  <-- this strictly follow KITTI raw data file format, while seperated into didi and kitti dataset. 
    ├── didi <-- will be something similar to kitti raw data format below. 
    └── kitti
        └── 2011_09_26
            ├── 2011_09_26_drive_0005_sync
            │   ├── image_02
            │   │   ├── data
            │   │   │   └── 0000000000.png
            │   │   └── timestamps.txt
            │   ├── tracklet_labels.xml
            │   └── velodyne_points
            │       ├── data
            │       │   └── 0000000000.bin
            │       ├── timestamps_end.txt
            │       ├── timestamps_start.txt
            │       └── timestamps.txt
            ├── calib_cam_to_cam.txt
            ├── calib_imu_to_velo.txt
            └── calib_velo_to_cam.txt
```



