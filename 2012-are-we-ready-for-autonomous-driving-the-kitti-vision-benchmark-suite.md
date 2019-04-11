|논문명 |Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite |
| --- | --- |
| 저자\(소속\) | Andreas Geiger\(\) |
| 학회/년도 | CVPR2012, [논문](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf) |
| 키워드 | |
| 데이터셋(센서)/모델 | |
| 관련연구||
| 참고 |[홈페이지](http://www.cvlibs.net/datasets/kitti/), [깃북설명](https://github.com/hunjung-lim/awesome-vehicle-datasets/tree/master/vehicle/kitti) |
| 코드 |[Download stereo 2015/flow 2015/scene flow 2015 data set (2 GB)](http://kitti.is.tue.mpg.de/kitti/data_scene_flow.zip) |


# The KITTI Vision Benchmark Suite

- stereo
- optical flow
- visual odometry / SLAM
- 3D object detection



## 1. Introduction

- 기존 benchmarks
- Caltech-101[17],
- Middlebury for stereo [41]
- optical flow [2]

![](https://i.imgur.com/AdIrsHK.png)
```
기존 데이터셋 vs KITTI데이터셋
```

## 2. Challenges and Methodology

- 2.1 실시간으로 데이터 수집 `collection of large amounts of data in real time`
- 2.2 칼리브레이션 `the calibration of diverse sensors working at different rates,`
- 2.3 GT 생성 `the generation of ground truth minimizing the amount of supervision required`
- 2.4 the selection of the appropriate sequences and frames for each benchmark
- 2.5 the development of metrics for each task.

### 2.1. Sensors and Data Acquisition

### 2.2. Sensor Calibration

### 2.3. Ground Truth

#### A. stereo and optical flow
- Given the **camera calibration**, the corresponding **disparity maps** are readily computed
- **Optical flow fields** are obtained by projecting the **3D points into the next frame**

#### B. visual odometry/SLAM

- The ground truth for visual odometry/SLAM is directly given by the output of the GPS/IMU localization unit projected into the coordinate system

#### C. 3D object

- 사람이 직접 진행
-

### 2.4. Benchmark Selection

### 2.5. Evaluation Metrics


#### A. stereo and optical flow
- we evaluate **stereo** and **optical flow** using the average number of **erroneous pixels** in terms of disparity and end-point error.
#### B. visual odometry/SLAM

#### C. 3D object

- 3단계로 구분 되어 있다. `Our 3D object detection and orientation estimation benchmark is split into three parts: `

##### 가. average precision (AP)

- First, we evaluate classical 2D object detection by measuring performance using the well established **average precision (AP)** metric as described in [16].

- Detections are iteratively assigned to ground truth labels starting with the largest overlap, measured by bounding box intersection over union.

- We require true positives to overlap by more than 50% and count multiple detections of the same object as false positives.

- We assess the performance of **jointly detecting objects** and estimating their **3D orientation** using a novel measure which we called the **average orientation similarity (AOS)**

![](https://i.imgur.com/KohYFCS.png)

- r = TP / (TP + FN) : the PASCAL object detection recall, where detected 2D bounding boxes are correct if they overlap by at least 50% with a ground truth bounding box. 

- The orientation similarity s ∈ [0, 1] at recall r is a normalized ([0..1]) variant of the cosine similarity defined as

![](https://i.imgur.com/3sXuOKi.png)

- D(r) denotes the set of all object detections at recall rate r 
- ∆^(i)_θ : the difference in angle between estimated and ground truth orientation of detection i. 

To penalize multiple detections which explain a single object, 
- we set δi = 1 if detection i has been assigned to a ground truth bounding box (overlaps by at least 50%) 
- δi = 0 if it has not been assigned.

##### 다. 3D object orientation estimation

- Finally, we also evaluate pure classification (16 bins for cars) and regression (continuous orientation) performance on the task of 3D object orientation estimation in terms of **orientation similarity**.

## 3. Experimental Evaluation

### 3.1. Stereo Matching

### 3.2. Optical Flow Estimation

### 3.3. Visual Odometry/SLAM

### 3.4. 3D Object Detection / Orientation Estimation

섹션 2.5에서 언급한 내용을 아래 세개의 항목에 대하여 `average precision`와 `average orientation similarity`를 평가 하였다. `We evaluate object detection as well as joint detection and orientation estimation using average precision and average orientation similarity as described in Sec. 2.5.`
- object detection 
- joint detection 
- orientation estimation


Our benchmark extracted from the full dataset comprises 12, 000 images with 40, 000 objects. 

We first subdivide the training set into 16 orientation classes and use 100 non-occluded examples per class for training the part-based object detector of [18] using three different settings: We train the model in an unsupervised fashion (variable), by initializing the components to the 16 classes but letting the components vary during optimization (fixed init) and by initializing the components and additionally fixing the latent variables to the 16 classes (fixed).


We evaluate all non- and weakly-occluded (< 20%) objects which are neither truncated nor smaller than 40 px in height. 

We do not count detecting truncated or occluded objects as false positives. 

For our object detection experiment, we require a bounding box overlap of at least 50%, results are shown in Fig. 6(a). 

For detection and orientation estimation we require the same overlap and plot the average orientation similarity (Eq. 5) over recall for the two unsupervised variants (Fig. 6(b)). 

Note that the precision is an upper bound to the average orientation similarity.

Overall, we could not find any substantial difference between the part-based detector variants we investigated. 

All of them achieve high precision, while the recall seems to be limited by some hard to detect objects. 

We plan to extend our online evaluation to more complex scenarios such as semi-occluded or truncated objects and other object classes
like vans, trucks, pedestrians and cyclists. 

Finally, we also evaluate object orientation estimation. 

We extract 100 car instances per orientation bin, using 16 orientation bins. 

We compute HOG features [12] on all cropped and resized bounding boxes with 19 × 13 blocks, 8×8 pixel cells and 12 orientation bins. 

We evaluate multiple classification and regression algorithms and report average orientation similarity (Eq. 5). 

Table 3 shows our results. 

We found that for the classification task SVMs [11] clearly outperform nearest neighbor classification. 

For the regression task, Gaussian Process regression [36] performs best.


