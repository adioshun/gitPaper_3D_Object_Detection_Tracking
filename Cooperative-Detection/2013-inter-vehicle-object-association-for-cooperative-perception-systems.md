|논문명|Inter-Vehicle Object Association for Cooperative Perception Systems|
|-|-|
|저자(소속)|Andreas Rauch (BMW)|
|학회/년도| ITSC 2013, [논문](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6728345)|
|키워드|협조탐지,Auction-ICP algorithm |
|참고||
|코드||


|년도|1st 저자|논문명|코드|
|-|-|-|-|
|2013|Andreas Rauch|Inter-Vehicle Object Association for Cooperative Perception Systems||
|2012|Andreas Rauch|Car2X-based perception in a high-level fusion architecture for cooperative perception systems||
|2011|Andreas Rauch|Analysis of V2X communication parameters for the development of a fusion architecture for cooperative perception systems||







# [Inter-Vehicle Object Association for Cooperative Perception Systems](https://ieeexplore.ieee.org/abstract/document/6728345)





협조 인지 시스템이란? In cooperative perception systems, different vehicles share object data obtained by their local environment perception sensors, like radar or lidar, via wireless communication. Inaccurate self-localizations of the vehicles complicate association of locally perceived objects and objects detected and transmitted by other vehicles. 

제안 내용 In this paper, a method for **intervehicle object association** is presented. 
- Position and orientation offsets between object lists from different vehicles are estimated by applying point matching algorithms. 
- Different algorithms are analyzed in simulations concerning their robustness and performance. 

결론 : Results with a first implementation of the socalled Auction-ICP algorithm in a real test vehicle validate the simulation results.

## I. INTRODUCTION


기존 프로젝트 `simTD` : Current research projects like simTD [1] try to exploit the benefits of wireless communication for advanced driver assistance systems. Therefor, every equipped vehicle and roadside station broadcasts its own position and dynamic state. 
- With this information, assistance systems like traffic light assistance, local hazard warning or cross traffic assistance can be realized. 

The project initiative Ko-FAS [2] with its joint project Ko-PER tries to enhance the scope of communication-based assistance systems by providing driver assistance systems a global view of the traffic environment.
- For this reason, every equipped vehicle and roadside station not only broadcasts its own position and state, but also a model of its dynamic environment based on its local perception sensors. 
- Thus, each equipped vehicle and roadside station can help to enhance the field of view of the other ones.
- This technology, called cooperative perception, will empower new forms of driver assistance and safety systems.



### 1.1 문제점 

A major challenge for cooperative perception systems and communication-based assistance systems in general is the **association of data** from different sources. 

In a cooperative perception system, a vehicle receives information about dynamic objects from other equipped vehicles or roadside stations. 

챌린지 : 자신의 로컬 환경 모델과 수신된 정보를 합치기 위해서는 시공간 정합은 필수 적이다. `In order to combine these perception data with its own local environment model, temporal and spatial alignment is crucial. `

### 1.2 기존 해결 책 

이전연구에서 시공간 정합 문제를 다루었다. `In [3], a method for temporal and spatial alignment in a cooperative perception system is proposed. `

```
[3] A. Rauch, F. Klanner, R. Rasshofer, and K. Dietmayer, “Car2xbased perception in a high-level fusion architecture for cooperative perception systems,” in Proceedings of the IEEE Intelligent Vehicles Symposium (IV), june 2012, pp. 270 –275.
```

하지만 통신대상의 자차위치 탐지 성능에 따라 결과가 매우 의존적인 단점이 있다. `However, the results of spatial alignment very much depend on the quality of the communication partners’ self-localizations.`

- 고정 물체의 경우 컨트롤된 상황에서 GPS이용 하여 측정하면 된다. `For static entities like roadside stations, the position and orientation can be determined by highly accurate means like carrier-phase differential GPS systems under optimal conditions.`

- 그러나 차량의 경우는 GPS가 효율적이지 않다. `However, vehicles often suffer from localization errors caused for example by GPS outage or multi-path effects in urban environments.`

이 문제가 해결 되지 않으면 전체결과도 나쁘게 된다. As soon as the self-localizations of the vehicles become deficient, the relative position and orientation of transmitted objects in the host vehicle’s reference frame become inaccurate, too. 

Such errors can have a strong negative impact on the quality of object association and fusion. 

Furthermore, there is no way to correct the state of objects which are only seen by the sender. 

### 1.3 본 논문의 목적 

논 논문의 목적은 불확실한 자차위치로 인한 부정적 영향을 최소화 하는것이다. `The goal of this paper is to reduce the negative impact of inaccurate self-localizations on the association and fusion quality. `

Therefore, an approach which aims to eliminate the bias between transmitted and local objects by estimating their optimal alignment is developed.

This task is carried out by point matching algorithms like the popular Iterative Closest Point (ICP) algorithm in this paper.


### 1.4 논문의 구성 

Subsequently, the paper is structured as follows: 
- Section II describes a high-level fusion architecture for cooperative perception. 
- In Section III, the concept of inter-vehicle object association using point matching algorithms is outlined. 
- In this context, special requirements of inter-vehicle object association and different categories of point matching algorithms are described. 
- Section IV provides a performance analysis of different point matching algorithms based on Monte Carlo simulations. 
- Experimental results with a first implementation of the Auction-ICP algorithm in a real test vehicle are presented in section V. 
- In Section VI, conclusions are drawn and future work is presented.


## II. HIGH-LEVEL FUSION ARCHITECTURE FOR COOPERATIVE PERCEPTION


In automotive applications, different architectures for sensor data fusion have been studied in the past. 

For distributed sensor networks, as in cooperative perception systems, a high-level fusion architecture is preferable due to its reduced communication bandwidth and its modularity. 

In [4], such a fusion architecture for cooperative perception systems has been introduced. 

This architecture is briefly described as follows.

![](https://i.imgur.com/l2IttMP.png)

Figure 1 illustrates the proposed architecture for a cooperative perception system within the host vehicle. 

2012논문 내용과 동일 (SKIP)
```
The fusion of the local perception sensors is performed within the local perception module which can also be based on a high-level fusion approach [5]. 

The result of this local fusion is an object list containing the states and corresponding covariance matrices, classification results and existence probabilities of the objects detected by the host vehicle’s local perception sensors.

The counterpart of the local perception is denoted as Car2X-based perception. 

In this module, communicated object data is prepared for later fusion with the output of the local perception. 

The temporal and spatial alignment according to the local perception’s reference frame is the major task of this module. 

As an output, an object list is passed to the global fusion module. 

The Car2X-based perception is based on messages like the cooperative awareness message (CAM) and the cooperative perception message (CPM). 

The latter either originates from a vehicle (v) or an infrastructure unit (i). 

The contents of these messages as well as further details of the proposed architecture for a cooperative perception system are described in [4]. 

A more detailed description of the Car2X-based perception module is provided in [3].
```

Both modules are supported with information about the position and dynamic state of the host vehicle by the ego data module.

In the global fusion module, objects detected by the host vehicle’s local perception and corresponding objects received from other vehicles or infrastructure units have to be associated. 

Appropriate association algorithms accounting for relative position and orientation errors of the object lists are presented in the remainder of this paper. 

After association, corresponding objects have to be fused in order to improve state estimation. 

As a result, the global fusion module provides a consistent global object list, which serves as input for the driver assistance system.


## III. INTER-VEHICLE OBJECT ASSOCIATION USING POINT MATCHING ALGORITHMS

### 3.1 Requirements of Inter-Vehicle Object Association

As the name implies, point matching algorithms are designed to match two sets of discrete points. 

However, in the automotive scenario of this work, two sets of extended objects must be matched, namely the objects perceived by the host vehicle and the objects perceived by another vehicle or infrastructure unit communicating with the host vehicle. 

For convenience, the latter will subsequently be denoted as sender. 

One would assume that it is possible to use the perceived objects’ centroids as input for matching. Unfortunately, if a tracked object is orientated in such a way that only one of its sides is exposed to the observing
vehicle’s sensors, neither its dimensions nor its centroid can be estimated correctly. 

To avoid this problem, we utilize instead the object corners which can be observed by both the host vehicle and the sender as a basis for point matching. 

Therefore, locally perceived objects as well as transmitted objects are converted to point clouds that consist of points in two dimensions which can be described as the four-tuple 




### 3.2 Point Matching Algorithms



클라우드 포인트 전체가 아닌 탐지된 객체의 x,y 중앙값 matching




## IV. PERFORMANCE ANALYSIS USING MONTE CARLO SIMULATIONS





