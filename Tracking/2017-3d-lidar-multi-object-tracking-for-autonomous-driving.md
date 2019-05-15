[3D-LIDAR Multi Object Tracking for Autonomous Driving](https://www.slideshare.net/adioshun/3dlidar-multi-object-tracking-for-autonomous-driving-111277160): A.S. Abdul Rachman,석사 학위 논문, 140page



# 

## 1. Introduction



### 1.3 Multi-Object-Tracking: Overview and State-of-Art

![](https://i.imgur.com/cMfis3G.png)

> tracking-by-detection 모델(탐지와 추적이 연속적으로 이루어짐)


#### On the detection stage, 

segmentation of the raw data is done to build basic feature of the scanned objects and to distinguish between dynamic and static objects. 

Subsequently, the segmented objects pose is then estimated based on either its outlier feature or fitting the measurement into known model[28]. 

At this stage, the raw measurement has already been translated into refined measurement with meaningful attributes (e.g. the static/dynamic classification and the objects pose). 

In another word, the object is successfully detected. 


Subsequently, the detected object is given to state estimation filter so that object kinematic states can be predicted according to a dynamic motion model. 

The purpose of the tracker is to effectively estimate the possible evolution of the detected object states even in the presence of measurement uncertainties, an optimal Bayesian filter such as Kalman Filter and Particle Filter are extensively used to address state uncertainties due to unmodelled dynamics of target evolution or noise acting on measurement.



## 2. data association (DA)

Next, a data association (DA) procedure is done by assigning detected object into a track with established filter state or newly started trajectory. 

The purpose of DA is to ensure the detected objects are localized while simultaneously maintaining their unique identity. 

At this stage uncertainties of the measurement due to finite sensor resolution and/or detection step imperfection may arise 

In order to address this issue, Bayesian probabilistic approaches are commonly used to handle DA process
- Joint Probabilistic Data Association Filter (JPDAF)[29] 
- Multiple Hypothesis Tracking (MHT)[30].


## 3. 

Finally, in practice a track management is required to cancel out spurious track based on specific criteria. 

Track management is responsible for handling existence and classification uncertainties based on a sensor-specific heuristic threshold. 

For example, track existence is defined to be true if 90% of its trajectory of track hypothesis is associated with a highly correlated measurement. 

The same also applies with class uncertainties; object class can be determined based thresholding of dimension, past motion, or visual cue (i.e. colour). 

Alternatively, these two uncertainties can also be assumed to be a random variable evolving according to Markovian
process. 

The Integrated Probabilistic Data Association[31] specifically model the existence of track as binary Markov process.

## 추가 팁 

In a typical urban scenario, sensor occlusions are unavoidable occurrences, to address this several works try to explicitly model occlusion[32, 33, 34] during detection and incorporated occlusion handling in their object tracking method. 

In the inter-process level, one notable approach as proposed by Himmelsbach[25] is the bottom-up/top-down approach which lets the tracker and detector exchange stored geometrical information to refine the detection parameter.

Alternatively, some other works have proposed a track-before-detect method that avoids explicit detection that takes sensor raw data as an input for tracking process[35].

> 여러 Lidar를 이용한 front-back/back-front 기법은 어떤가? 


---
## 2. Detection Fundamentals

### 2.1 Overview 

### 2.2 Spatial Data Acquisition using 3D LIDAR

### 2.3 Segmentation

A large amount of point clouds data demand a high computational power to process, in addition,
due to discontinuous nature of point cloud, it is useful to combine the geometric points into a
semantically meaningful group. Therefore, the raw measurement needs to be pre-processed to
eliminate unnecessary element and reduce the dimensionality of possible target object before
it passed to the detection process. The segmentation process mainly deals with differentiating
non-trackable objects such as terrain and kerb from unique objects of interest such as cars,
cyclists and pedestrians.

Segmentation method can be divided into two groups based on the underlying tracking
scheme[28]: the grid-based and the object-based. The grid-based segmentation is mainly
used for track-before-detect scheme, and the object-based is used mostly for the track-by
detect scheme. Although it is important to note the relation is not exclusive. For example,
Himmelsbach et al.[25] used grid-based approach in the pre-processing stage (specifically
clustering) but later used object-based approach to perform tracking.

#### A. Grid-based
The grid cell-based methods chiefly rely on the global occupancy grid maps to indicate the
probability of an object existing in a specific grid (i.e. occupied). One common approach to
update the probabilities is to compare the occupancy in current time frame k with occupancy
in time frame k − 1 with the Bayesian Occupancy Filter[74]. In some implementation, Particle
Filter is also used in used derive velocity in each grid[75, 76]. The particles with positions
and velocities in a particular cell represent its velocity distribution and occupancy likelihood.
When the neighbouring cells with similar speeds are clustered, then each cluster can be
represented in the form of an oriented cuboid. Grid-based tracking results in simpler detection
process and less complicated data association process, however, the disadvantage of grid-based
representations is if a moving object cannot be detected (e.g. due to occlusion), the area
will be mapped as a static grid by default if no occlusion handling is applied. Additionally,
grid-based representations contain a lot of irrelevant details such as unreachable free space
areas and only have limited ability to represent dynamic objects[36]. The latter part suggests
grid-based detection is insufficient for the purpose of urban autonomous vehicle perception
which require detailed representation of dynamic objects to model the complex environment.
Therefore, we shall focus on the track-by-detect scheme, and the object-based detection will
be explored more in-depth.

#### B. Object-based
Object based segmentation on the other hand, uses the point model (or rather collections of
bounded points) to describe objects. Unlike grid-based method, a separate pose estimation
and tracking filter are required to derive dynamic attributes of the segmented object. The
segmentation is chiefly done with ground extraction to separate non-trackable object with
objects of interest, followed by clustering to reduce the dimensionality of tracked objects. The
two steps will be discussed in the following subsections. Note that the step-wise results of
object-based segmentation processes can be seen in Figure. 2-5.


##### 가. Ground Extraction
Due to non-planar nature of roads, a point cloud coming from 3D Laser scanner also includes
terrain information which is considered as non-obstacle (i.e. navigable) by the vehicle. It is
useful to semantically label the navigable terrain (hereby called ground) from the elevated
point that might pose as an obstacle. Ground extraction is an important preliminary process
in object detection process. As we are going to deal with a large number of raw LIDAR
measurement, computation load must be factored during implementation. Chen et al.[77]divide ground extraction process into three subgroups: grid/cell-based methods, line-based
methods and surface-based methods. Meanwhile, Rieken et al.[15] consider line-based and
surface-based method as one big group called scan-based method. Grid-cell based method
divides the LIDAR data into polar coordinate cells and channel. The method uses information
of height and the radial distance between adjacent grid to deduct existence of obstacle when
slope between cell cross certain threshold, i.e. slope-based method (see[12, 10]).
On the other hand, scan-based method extracts a planar ground (i.e. flat world assumption)
derived from specific criteria, one of the approaches is to take the lowest z value and applying
Random sample consensus (RANSAC) fitting to determine possible ground[78]. The advantage
of grid cell-based method is that the ground contour is preserved and flat terrain is represented
better. However, compared to the scan-based method it does not consider the value of neig-
hbourhood channels, and thus may lead to inconsistency of elevation due to over-sensitivity to
slope change. Ground measurement may also be incomplete due to occlusion by a large object.
One notable approach which factored the occlusion is by Rieken et all.[14], they combine of
channel-wise ground classification with a grid-based representation of the ground height to
cope with false spatial measurements and also use inter-channel dependency to compensate
for missing data.

##### 나. Clustering
A large number of point clouds are computationally prohibitive if the detection and tracking
are to be done over individual hits. Therefore, these massive point cloud is to be reduced
into smaller clusters in which each of them is simply a combination of multiple, spatially
close-range samples; the process is called clustering. Clustering can be either done in 3D, 2D
(taking the top-view or XY plane) or 2.5D[79] which retain some z-axis information such as
elevation in occupancy grids.
2D clustering offers computationally simpler operation. Rubio et al.[80] presented a 2D clus-
tering based on Connected Component Clustering which has shown to be implementable in
real-time due to its low complexity, this method is also used in 3D object tracking method by
Rieken et al.[14].
Some literature in 2D object tracking[25, 45, 81] have shown that this approach is often
sufficient in the application of object tracking. However, care should be taken as vertically
stacked objects (e.g. pedestrian under a tree) will be merged into one single cluster, which
might be undesirable depending on the vehicle navigation strategy.
3D clustering offers high fidelity object cluster that incorporates the vertical (z-axis) features.
Still, the resulting data and the computational effort required is some magnitude larger than
its 2D counterpart. Compared to 2D clustering, there are fewer works which explicitly deal
with 3D clustering for object tracking with LIDAR data. Klasing et. al.[63] proposed a 3D
clustering method based on Radially Bounded Nearest Neighbor (RNN), and more recently
Hwang, et. al[82] using DBSCAN (Density-Based Spatial Clustering of Applications with
Noise) to employ full 3D clustering. Considering real-time requirement and significant number
of points involved, 2 or 2.5D clustering is more preferred[83] owing to the fact vehicle onboard
computer likely to have limited computational power.

### 2.4 Pose Estimation

물체의 경로나 방향을 알기위해서는 [자세추정]이 진행 되어야 한다. `Subsequently, in order to extract usable information in term of the object trajectory and orientation, the pose estimation needs to be done. `

물체의 자세(pose)란 다음의 것들을 의미 한다. `Object pose is a broad term that may include the`
- dimension, 
- orientation (heading), 
- velocity and 
- acceleration of such objects. 

분류 `Pose estimation generally can be grouped into model-based or feature-based method. `
- 모델 기반 : 센서 수집값과 알고 있는 도형 모델 연결 `Model-based pose estimation aims to match raw measurement into a known geometric model, `
- 특징 기반 : 물체의 특징 정로보 추론 `while feature-based pose estimation deduces object shape from a sets of feature.`


#### A. Model-based

##### 가. 모양과 관련하여 

모델 기반 방식은 최적화 기법을 이용하여 반복하여 차량을 cuboid등에 맞춘다. `Model-based pose estimation uses optimization-based iteration to fit vehicle into cuboid or rectangle representation. `

The cuboid object is parametrized and the most probable vehicle pose from the segment points are iterated. 

In order to fit clusters of points into a model, edge-like features are to be extracted and "best-fit" method is utilised to fit it into a known model.


Barrois[84]
- A notable example is in Barrois[84] where the optimization problem is formulated as the minimization of the polar distance between the scene points obtained and the visible sides to compute the best vehicle pose.

Petrovskaya and Thurn[21]
- Petrovskaya and Thurn[21] use importance sampling scoring based on the fitting of measurement to a predetermined geometric model. 

Another interesting approach is by Morris et.al.[85] 
- whose matched filter takes view-dependent self-occlusion effects into account, 
- and utilise 4 rectangles to represent the inner and outer sides of the vehicle.

##### 나. 방향과 관련하여 

Another the major challenge in bounding box generation is the orientation estimation, 

일반적 방법은 : common approaches are by calculating the minimum area of clustered points[86], 

단점은, 가려진 부분이 있으면 성능이 않좋다. however in the presence of partial occlusion the results can be spurious in term of dimension and orientation accuracy, 

문제 해결 #1 : to tackle this problem Rieken, et al.[43] uses an L-, U- or I-like simple set of geometric classifier to derive most appropriate orientation. 

문제 해결 #2 : An alternative approach is to use convex-hull method to generate bounding box[87, 52]; the idea is to minimize the average distance between the
convex-hull points and fit a rectangle. (see Figure 2-6).



![](https://i.imgur.com/XcTXYMv.png)

Model-based pose estimation can also be combined with Bayesian probabilistic likelihood such as in the works of Vu and Aycard[88], Liu[64], and Nashashibi[89]. 

They explicitly modelled the possible occlusion area to estimate the vehicle dimension based on scan-line distance.

Liu[64] in particular uses "transitional region" between the inner-outer bounding box model object aiming to accommodate more measurement errors.


모델 기반의 단점은 
- 계산 시간이 큼 `Although model-based method offers optimal pose estimation, the major disadvantage of this method is the high computational time required, and this may not be suitable for realtime application. `
- 최적화 측면에서 Local Minima에 빠질 가능성이 있다. `Moreover, the optimisation problem may reach a solution at local minima depending on the initialization and results in a sub-optimal pose. `

그래서 특징 기반 방식이 선호 된다. `As a consequence, the feature-based pose estimation shall be preferred.`


#### B. 



---

## 3. Tracking Fundamentals

### 3.1 Overview

![](https://i.imgur.com/EW4cpdY.png)

1. The result of detection is used to start a new track, or if existing track exists, the measurement is checked
if it statistically likely to be correct measurement through gating. 

2. Passing the gating is the prerequisite of association between measurement and tracks before being sent forward to state
estimator. 

This chapter shall cover the assumed modelling of sensor and target dynamics, used not only in state estimatio and prediction, but also Data Association. 

Accordingly, several classes of Bayesian filter will be introduced.


### 3.2 Sensor and Target Modelling

The LIDAR sensor is placed on moving ego-car, which is considered as the origin. 

Due to the measuring principle of rotating LIDAR sensor the measurement originally comes in polar coordinates (in the form of distance and bearing). 

However, Velodyne digital processing unit readily provides the sensor measurement on the Cartesian coordinate system, 

In this thesis, the latter coordinate system will be used to conform better with ego-vehicle navigation frame. 

The relation between the ego-car navigation frame and sensor measurement frame can be seen in Figure 3-2.


> ???

### 3.3 Object Tracking as A Filtering Problem

In this thesis, object tracking problem is modelled as a filtering problem in which the object states are to be estimated from noisy, corrupted or otherwise false measurements. 

The estimated states and assumed system dynamics are given in the previous section. 

The basic concept of Bayes filtering is introduced along with the filters which will be used during implementation.


#### A. Bayes Theorem in Object Tracking


#### B. Unscented Kalman Filter

#### C. Interacting Multiple Mode


#### D. Data Association Filter

칼만필터가 예측값의 최적화를 지원 하긴 하지만 `KF (or its variant) offers the ability to provide optimal estimates of an object track. `

그럼에도불구하고  센싱의 불확실성으로 인해서 추적 물체가 해당 물체가 맞는지 보장 할수 없다. `Notwithstanding, due to measurement uncertainty there is no guarantee that the tracked object is actually a relevant object, or even if it is an existing object in the first place. `

따라서 후속적인 분류및 확인 절차가 필요 하다. `Therefore, a subsequent classification and validation of the estimated track are simply necessary.`

Data Association (DA) is a process of associating the detection result into a tracking filter.

There are two classes of DA filter: 
- the deterministic filter and 
- the probabilistic filter. 


##### 가. NNF

- Representative of deterministic DA filter is Nearest Neighborhood Filter (NNF) algorithm 

- NNF updates each object with the closest measurement relative to the state. 

- NNF associates object with known track based on the shortest Euclidean or the Mahalanobis distance between the
measurement and track.

##### 나. PDAF

- The probabilistic DA filter that is very well-known in object tracking literature body is the
eponymous Probabilistic Data Association Filter (PDAF)[29]. 

- The PDAF perform a weighted update of the object state using all association hypotheses in order to avoid hard, possibly erroneous association decisions commonly encountered in the use of NNF algorithm. 

- The erroneous association is often found during the scenario in which multiple measurements is located close to each other (i.e. clutter) and results in single measurement being used to incorrectly update all other nearby objects.


> PDA is also one of the most computationally efficient tracking algorithms among clutter-aware tracker[97], for
instance when compared to MHT[117]. 


![](https://i.imgur.com/eJBHCLO.png)



### 3.4 Probabilistic Data Association Filter (단일)


### 3.5 JPDA: Tracking Multiple Target in Clutter (멀티)









