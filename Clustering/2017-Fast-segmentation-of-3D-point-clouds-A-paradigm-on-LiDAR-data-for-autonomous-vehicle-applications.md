


# [Fast segmentation of 3D point clouds: A paradigm on LiDAR data for autonomous vehicle applications](https://www.mendeley.com/viewer/?fileId=1e8417cb-a82c-1ef7-6ed2-4e9e076b2602&documentId=e4b0a84b-1900-3a22-a2d3-bad599279302)

> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989591

[[ROS] Run_based_segmentation](https://github.com/VincentCheungM/Run_based_segmentation), [[바닥제거]](https://github.com/lorenwel/linefit_ground_segmentation)



Abstract— 자율주행차를 위해서 세그멘테이션이 중요하다. `The recent activity in the area of autonomous vehicle navigation has initiated a series of reactions that stirred the automobile industry, pushing for the fast commercialization of this technology which, until recently, seemed futuristic. The LiDAR sensor is able to provide a detailed understanding of the environment surrounding the vehicle making it useful in a plethora of autonomous driving scenarios. Segmenting the 3D point cloud that is provided by modern LiDAR sensors, is the first important step towards the situational assessment pipeline that aims for the safety of the passengers. `

세그멘테이션은 지표면 분리, 장애물 분리 절차가 필요 하다. `This step needs to provide accurate segmentation of the ground surface and the obstacles in the vehicle’s path, and to process each point cloud in real time. The proposed pipeline aims to solve the problem of 3D point cloud segmentation for data received from a LiDAR in a fast and low complexity manner that targets real world applications. `

The two-step algorithm 
- first extracts the ground surface in an iterative fashion using deterministically assigned **seed points**, 
- and then clusters the remaining non-ground points taking advantage of the structure of the LiDAR point cloud. 

Our proposed algorithms outperform similar approaches in running time, while producing similar results and support the validity of this pipeline as a segmentation tool for real world applications.

## I. I NTRODUCTION

자율 주행차에 대한 관심이 크다. 센서 퓨전을 통한 안정성 확보가 중요 `The recent activity in the area of autonomous vehicle navigation has initiated a series of reactions that stirred the automobile industry, pushing for the fast commercialization of this technology which, until recently, seemed futuristic. Despite the excitement of the general audience, it is imperative for the engineering community to realise the responsibility of bringing such product to a mass production level. This realization has pushed for the fusion of multiple sensors in order to enhance the sensing capabilities of the autonomous vehicles.`

3D 라이다 센서도 중요한 센서중 하나이다. `One such sensor is the LiDAR, which utilizes multiple laser beams to locate obstacles in its surroundings and is known for its capability to depict this information in a dense three dimensional (3D) cloud of points. The LiDAR has been popular amongst academic research teams for its long range and satisfactory accuracy while recent hardware advancements that promise better, lower cost, and smaller scale sensors have appeared to attract the interest of the industry.`

3D 라이다 센서로 장애물 탐지, 속도, 분류를 알아 내는것이 중요 하다. `Mounted on an autonomous vehicle the sensor by itself provides the means to acquire a 3D representation of the surrounding environment, and the challenge is to analyze it and extract meaningful information such as the number of obstacles, their position and velocity with respect to the vehicle, and their class being a car, a pedestrian, a pole, etc. `

이를 위해 중요한 첫 단계는 세그멘테이션이다. `Similar to image processing, the first important step for this type of analysis is the fine segmentation of the input data into meaningful clusters. The work in this paper is attacking this exact problem and is presenting a methodology that focuses on computational speed and complexity. A fast and low complexity segmentation process allows to redirect precious hardware resources to more computationally demanding processes in the autonomous driving pipeline.`

**[주요내용]** 라이다로 수집되는 데이터는 점군이며, 각 Layer에 있는 점들은 타원형 형태이다. 또하 타원형 Layer의 시작 포인트는 같은 방향(Orientation)을 향하고 있다. `In the case of LiDAR sensors with the ability to capture 360 degrees of information, the data is represented as a set of 3D points called a point cloud which is organized in layers. The points in each layer are also organized in an elliptical fashion and the starting points of all elliptical layers are considered to share the same orientation.`

제안 방식은 위 특성을 이용 효율적인 인덱싱을 통해 세그멘테이션 한다. `The presented methodology relies on this type of organization in the point cloud and takes advantage of smart indexing to perform an efficient segmentation. `

비슷한 이전 연구와 같이 제안 방식도 두 스텝으로 되어 있다. `Similar to preceding work in the same domain, our approach proposes the segmentation process to conclude in two steps; `
- (i) 지표면 분류 ` the extraction of the points belonging to the ground, `
- 나머지 중 물체 분류 `and (ii) the clustering of the remaining points into meaningful sets. `

Both steps present original approaches to the problem focused on real world applications and extensively tested on
the publicly available KITTI dataset [1].

문서 구성 `In the following,`
-  the related literature review can be found in section II. 
- The methodology is explicitly described in section III 
- and results can be found in section IV, 
- followed by the last section V that summarises the findings of this work.


## II. LITERATURE REVIEW

바닥과 물체 등 세그멘테이션 관련 기존 연구 `In this work we are focusing on segmentation methods for data points in three dimensions and emphasize to applications related to autonomous vehicle driving. A common practise for this particular application is to split the segmentation process in two steps; first extracting the ground points and then clustering the remaining points into objects the car needs to be aware of. `

### 2.1 바닥제거 

For the first step, a grid-based approach was introduced by Thrun et al. [2] which are dividing the grid cells as ground and non-ground based on the maximum absolute difference between the heights of points inside the cell. 

On the other hand, Himmelsbach et al. [3] are treating the point cloud in cylindrical coordinates and taking advantage of the
distribution of the points in order to fit line segments to the point cloud. The segments, based on some threshold of the slope are considered to capture the ground surface. 

In an attempt to recognize the ground surface, Moosmann et al. [4] are creating an undirected graph and compare local changes of plane normals in order to characterize changes in the slope. 

Douillard et al. [5] are introducing GP-INSAC, a gaussian-process based iterative scheme that classifies points in ground and non-ground based on the variance of their height from the mean of a gaussian distribution. 

Similarly, gaussian processes for ground extraction were later used by Chen et al. [6], 

while probabilistic approaches utilizing Markov Random Fields can be seen in the work of Tse etal. [7] and Byun et al. [8].

### 2.2 물체 군집화 

이후 바닥에 제거된 나머지 포인트들에서 의미있는 그룹핑을 하는것은 군집화 문제로 다루어 졌다. `Consecutively, the grouping of the remaining well separated non-ground points is usually treated as a clustering problem where appropriate well known clustering algorithms are employed. Such are the cases for clustering algorithms that are popular for their simplicity, easy deployment, and high execution speed. `

Examples include the **euclidean cluster** extraction [9] whose implementation can be found in the  point cloud library (PCL) [10], DBSCAN [11], and Mean-Shift [12]. 

The use of **voxelization techniques** to compress and then cluster the remaining non-ground points is also considered popular ([3], [5]). 
- These algorithms traverse the point cloud in an irregular way and upon finding an unlabeled point, they assign a new label which is then propagated to neighboring unlabeled points based on some rules. 

Inside a three dimensional space, such irregular accessing of points can lead to exhaustive search for neighbors that slow down the whole process. Although this is necessary for unorganized point clouds, in the targeted application the layer-based
organization of the point cloud is not exploited.


## III. M ETHODOLOGY

본 장에서 자세한 내용을 언급 한다. `The following paragraphs describe in detail the complete methodology for the segmentation of a point cloud received by a 360 o coverage LiDAR sensor. `

먼저, GPF(`Ground Plane Fitting`)를 통해 바닥을 제거한다. 이후 제안 방식인 SLR(`Scan Line Run`)으로 클러스터링을 한다. SLR은 이미지 처리 알고리즘인 CCL(`connected components labeling`)에서 아이디어를 얻었다. `First, we present a deterministic iterative multiple plane fitting technique we call Ground Plane Fitting (GPF) for the fast extraction of the ground points, followed by a point cloud clustering methodology named Scan Line Run (SLR) which is inspired by algorithms for connected components labeling in binary images.`

Each paragraph is conceptually divided in three sections including a brief reasoning behind the algorithm selection
along with the definition of new terms, the overview of the algorithm according to the pseudocode diagrams, and
discussion of algorithm implementation details.

### 3.1 Ground Plane Fitting (바닥제거 방법) 

Cloud points that belong to the ground surface constitute the majority of the point cloud and their removal significantly reduces the number of points involved in the proceeding computations. The identification and extraction of ground points is rather suitable for this application for two main reasons; 
- (i) they are easily identifiable since they belong to planes, which are primitive geometrical objects with a simple mathematical model, 
- and (ii) it is acceptable to assume that points of the point cloud with the lowest height values are most likely to belong to the ground surface. 

This prior knowledge is used to dictate a set of points for the initiation of the algorithm and is eliminating the random selection seen in typical plane-fit techniques such as the RANdom SAmple Consensus (RANSAC), resulting in much faster convergence.

Generally, a single plane model is insufficient for the representation of the real ground surface as the ground points do not form a perfect plane and the LiDAR measurements introduce significant noise for long distance measurements.

We have observed that in most instances the ground surface exhibits changes in slope which need to be detected. 

The proposed ground plane fitting technique extends its applicability to such instances of the ground surface by dividing evenly the point cloud into a number of segments N segs along the x-axis (direction of travel of the vehicle), and applying the
ground plane fitting algorithm in each one of those segments. 

![](https://i.imgur.com/d6MUiox.png)

As depicted in the main loop of Alg. 1, for each of the point cloud segments the ground plane fitting starts by deterministically extracting a set of seed points with low height values which are then used to estimate the initial plane model of the ground surface. 

Each point in the point cloud segment P is evaluated against the estimated plane model and produces the distance from the point to its orthogonal projection on the candidate plane. 

This distance is compared to a user defined threshold Th_{dist} , which decides whether the point belongs to the ground surface or not. 

The points belonging to the ground surface are used as seeds for the refined estimation of a new plane model and the process repeats for N_{iter} number of times. Finally, the ground points resulting from this algorithm for each of the point cloud segments can be concatenated and provide the entire ground plane.

Our approach for the selection of initial seed points introduces the lowest point representative (LPR), a point defined as the average of the N_{LPR} lowest height value points of the point cloud. 

The LPR guarantees that noisy measurements will not affect the plane estimation step. 

Once the LPR has been computed, it is treated as the lowest height value point of the point cloud P and the points inside the height threshold Th_{seeds} are used as the initial seeds for the plane model estimation.

......

> 중략 

### 3.2 Scan Line Run

바닥제거후 남겨진 포인트는 군집화를 한다. 본 논문의 목적은 각 포인트들에 라벨값을 할당하여 세그멘테이션 하는것이다. `The remaining points P ng that do not belong to the ground surface need to form clusters to be used in higher level post processing schemes. Our goal is for each point p_k \∈ P_{ng} to acquire a label {i} that represents its cluster identity while using simple mechanisms that will ensure the fast running time and low complexity of the process.`

라이다의 멀티레이어 구조는 2D 이미지의 row-wise 구조와 유사하다. 차이점은 구성요소가 uneven라는 것과 서클 형태라는 것이다. `In the case of 360 o LiDAR sensor data, the multi-layer structure of the 3D point cloud resembles strongly the row-wise structure of 2D images with the main differences being the uneven number of elements in each layer and its circular form. `

제안 방식은 3D 포인트를 픽셀처럼 처리 하는것이다. 그리고 Two-Run CCL을 적용한다. `The proposed solution treats the 3D points as pixels of an image and adapts a two-run connected component labeling technique from binary images [13] to produce a real time 3D clustering algorithm.`

본 논문에서는 동일한 스캔링에서 생성되는 포인트들을 **Layer**라고 지칭한다. 그리고 그 구성 포인트들을 **run**이라고 지칭한다. `We call a layer of points that are produced from the same LiDAR ring a scan-line. Within each scan-line, its elements are organized in vectors of contiguous points called runs.`

run구성요소들은 같은 라벨을 가지고있다. 그리고 클러스터를 구성하는 메인 블록이다. `The elements within a run share the same label and are the main building blocks of the clusters. `

![](https://i.imgur.com/VVL04Fy.png)

According to Alg. 2 and without loss of generality, we assume the point cloud P_{ng} is traversed in a raster counter-
clockwise fashion starting from the top scan-line. 

The runs of the first scan-line are formed and each receives its own new Label which is inherited by all of its point-elements.

The runs of the first scan-line then become the runs Above and are used to propagate their labels to the runs in the subsequent scan-line. 

The label is propagated to a new run, when the **distance** between a point of the new run and its nearest neighbor in the above scan-line is less than T_h merge .

When many points in the same run have nearest neighbors with different inheritable labels, the winning label is the
smallest one. 

On the other hand, when no appropriate nearest neighbors can be found for any of the points in the run, it receives a new Label. 

The above are performed in a single pass through the point cloud and when this is done, a second pass is performed for the final update of the point’s labels and the extraction of the clusters.

![](https://i.imgur.com/f7saWYD.png)

흰색원 바닥, 색깔원 물체 `The following example accompanying Fig. 1 covers the main instances of the proposed algorithm with the white and colored circles representing ground and non-ground points respectively. `

파란점은 아직 세그멘테이션이 진행 되지 않았음을 의미 한다. `The blue circles are non-ground points not yet visited. `
##### Fig1. Step A
In step a), 초기화 후 두개의 RUN(오렌지/그린)이 라벨 1,2를 할당 받음 `the first scan-line is initialized with two runs (orange and green) each receiving a newLabel (1 and 2 inside the triangles)`. 

##### Fig1. Step B
Step b) 새로운 라벨 생성이나, 라벨 전달에 대한 설명 `demonstrates the assignment of a newLabel and the propagation of two labels. `
- 8번과 가장 가까운 것은 2번이지만 거리가 멀어 8번은 **새로운 라벨 3**을 할당 받아`In particular, the nearest non-ground neighbor of 8 is 2 and their distance is greater than Th_{merge} . In this case, {labelsToMerge} is empty and point 8 represents a new cluster. `
- 반면, 10번과 가장 가까운 것은 3번이고 거리도 가까우므로 **기존 라벨 1**을 전달 받음 `On the other hand, the nearest non-ground neighbor of 10 is 3 with their distance smaller than Th_{merge} , which makes label 1 to propagate over to point 10. `
- 비슷한 방법으로 12,13번도 5,6번과 거리가 가까으므로 **기존라벨 2**을 전달 받음 `Similarly, points 12 and 13 are both close to their respective neighbors 5 and 6, and based on the non-empty {labelsToM erge}, label 2 is assigning to them. `

##### Fig1. Step C
하나의 라벨을 받았지만, 다른 RUN이 도달 하였을 경우 `Next, the final scan-line is considered in step c) where one run is present. `
- 두개의 라벨과 거리가 가까울 경우 가장 작은 값(1 Vs. 2) 을 할당 한다. `Points 17 and 19 have neighbors 10 and 12 which belong to different clusters and are both appropriate to propagate their label. According to our algorithmic logic the smallest of the two labels (namely label 1) is inherited.`

##### Fig1. Step D
두 라벨이 합쳐진 경우 `In step d) the merging of the two labels 1 and 2 is noted and handled accordingly by the label equivalence resolving technique which is discussed below.`


#### Implementation Details

방법은 간단 하지만 효율적 구현을 위해 아래 방법을 제안 한다. `The outline of the algorithm is straight forward, but for an efficient implementation we propose solutions on `
- (i) how to create runs, 
- (ii) how to look for the nearest neighbor, 
- and (iii) how to resolve label conflicts when merging two or more connected components.

##### 1. how to create runs, 
i) A run is created upon the first visit of the scan-line
as a vector of indeces and holds information on which
consecutive points are close enough to be considered a single
block inside a scan-line. Considering the circular form of the
scan-lines, a run may bridge over the first and last indeces.
When detected, this case is resolved by attaching the indeces
of the ending of the scan-line at the beginning of the indeces
of the first run as seen in the example of Fig. 3.

##### 2. how to look for the nearest neighbor, 
ii) When the input point cloud is expressed in cylindrical
coordinates with points x = [r θ z], then indexing the
nearest neighbor in the scan-line above can be viewed as
simply comparing θ values. In autonomous vehicle applica-
tions though, clustering is one small component of a much
larger system of sensors and algorithms, and the cartesian
coordinate system is preferred for compatibility reasons.
Implementation-wise, the naive solution is to build a kdtree
structure with all the non-ground points in the scan-line
above and use this to find each nearest neighbor, resulting in
a suboptimal but viable solution that can be further refined.
Under the assumption that the points in a scan-line are
evenly distributed along the whole scan-line, we are utilizing
a smart indexing methodology that overcomes the problem
of the uneven number of elements in the different scan-
lines and significantly reduces the number of queries for the
nearest neighbor. Assume that each scan-line has N i number
of points and that each point owns two indeces; one global
ind g which represents its position in the whole point cloud,
and one local ind l that identifies the point inside the scan-
line. One can easily alternate between the indeces of the
scan-line K by:

> 중략 

##### 3. how to resolve label conflicts

iii) The methodology to resolve label merging conflicts is
being introduced in [13] where all the details for the imple-
mentation and deep understanding are provided.

> 중략 


## IV. E XPERIMENTAL R ESULTS

### A. Ground Plane Fitting

### B. Scan-Line Run

## V. CONCLUSIONS AND FUTURE WORK

We have examined the problem of segmenting point cloud data for applications in autonomous driving vehicle and have proposed a pipeline that initially extracts the ground points and consequently groups the remaining points into clusters based on their distance. 

The proposed solution is tailored around the specific application and performs significantly faster than general purpose segmentation pipelines, which makes it ideal for real time operations on data gathered by LiDAR sensors.

The algorithm has been successfully tested in several sequences of the publicly available KITTI dataset and its performance has been verified for a plethora of different scenes and number of points. 

향후 연구 : An additional step is to verify how the pipeline behaves when encountering rough, uneven terrain and rapid slope changes. 
- For this we plan to collect our own data that will capture corner cases and will allow us to verify the robustness of our algorithms.

Segmentation is the first step in the autonomous scene understanding. Once the detection of the obstacles in the environment surrounding the vehicle is captured, the next processing step is to identify the obstacles as static or
dynamic and perform tracking on the dynamic ones.
