# Car2X-based perception in a high-level fusion architecture for cooperative perception systems

https://ieeexplore.ieee.org/document/6232130



In cooperative perception systems, different vehicles share object data obtained by their local environment perception sensors, like radar or lidar, via wireless communication. 

In this paper, this so-called Car2X-based perception is modeled as a virtual sensor in order to integrate it into a highlevel sensor data fusion architecture. 

The spatial and temporal alignment of incoming data is a major issue in cooperative perception systems. 

시간 정렬:Temporal alignment is done by predicting the received object data with a model-based approach. 
- In this context, the CTRA (constant turn rate and acceleration) motion model is used for a three-dimensional prediction of the communication partner’s motion. 

공간 정렬: Concerning the spatial alignment, two approaches to transform the received data, including the uncertainties, into the receiving vehicle’s local coordinate frame are compared. 
- The approach using an **unscented transformation** is shown to be superior to the approach by **linearizing the transformation** function. 
- Experimental results prove the accuracy and consistency of the virtual sensor’s output.


## I. INTRODUCTION

simTD프로젝트에서는 차량과 도로변 장비들이 자신의 위치와 dynamic state를 브로드캐스트 한다. 이를 이용하여 ADAS를 구현한다. `Current research projects like simTD [1] try to exploit the benefits of wireless communication for advanced driver assistance systems. For this reason, every equipped vehicle and roadside station broadcasts its own position and dynamic state. With this information, assistance systems like traffic light assistance, local hazard warning or cross traffic assistance can be realized.`

Ko-FAS프로젝트에서는 simTD의 정보 외에도 인지센서를 통해 습득한 model of its own dynamic environment로 전송한다. 이를 이용하여 다른 차량의 FoV를 증대 시킨다. `The project initiative Ko-FAS [2] with its joint project Ko-PER tries to enhance the scope of communication-based assistance systems by providing driver assistance systems a global view of the traffic environment. For this reason, every equipped vehicle and roadside station not only broadcasts its own position and state, but also a model of its own dynamic environment based on its local perception sensors. Thus, each equipped vehicle and roadside station can help to enhance the field of view of the other ones.`

이러한 기술들은 **cooperative perception**라 불리운다. `This technology, called cooperative perception, will empower new forms of driver assistance and safety systems.`

As mentioned above, past research in the field of Car2X communication focused on the transmission and use of the state of the communication partners, also referred to as sender vehicles, within the perception framework of the host vehicle. 

### 1.1 이전연구 [3]

이전 연구 [3]에서는 이웃차량의 상태와 호스트차량의 상태를 EKF를 사용하여 글로벌 좌표로 예측하여 사용하였다. `In [3], the state of the sender vehicle as well as the state of the host vehicle are predicted to the current time in global coordinates using an extended Kalman filter and a turn model. `
- 이후 두차량의 글로벌 상태 정보는 호스트 차량의 로컬 좌표계로 이웃차량의 relative state로 변환하였다. `After that, the global states of both vehicles are transformed into a relative state of the sender vehicle in the host vehicle’s local coordinate frame. `
- relative state의 covariance를 계산하기 위하여 **linearization of the transformation**이 사용되었다. `For the computation of the covariances of the relative state, the linearization of the transformation is employed. `
- 이 데이터는 로컬센서의 결과값과 association하기 위해 사용되었다. `This data is used for association with the output of the local sensors. `
- 성공적으로 association되었다면 로컬에서 인지된 물체는 통신으로 받은 데이터를 이용하여 크기나 class정보가 보정 되었다. `In case of a successful association, the locally perceived object is complemented by additional data from the wirelessly communicated object, like object size and class. `
- 평가를 위해서 **correct association rate**값을 사용 하였다. `For the evaluation of the system, the correct association rate is used.`

### 1.2 이전연구 [4]

> DGPS / EKF를 이용 조향, 속도 모션 모델 보정 

In [4], the communicated state from a sender vehicle is also predicted to the current time. 

Similar to the work mentioned above, an extended Kalman filter using a constant turn rate and acceleration (CTRA) motion model is employed. 

This time, the two-dimensional prediction is done in a horizontal local coordinate frame. 

For efficient fusion with object data obtained by a local radar sensor, the covariance matrices of the communicated object data and the ones obtained by the radar sensor are used as tuning parameters. 

For the evaluation, the DGPS positions of the communicating vehicles are assumed as ground truth.


### 1.3 이전연구 [5]

> 교차로에 위치한 인지시스템의 정보를 차량이 활용 

In INTERSAFE-2 [5], a European research project, another preparing leap towards Ko-PER’s vision is made.

Infrastructure units as an additional information source are equipped with perception sensors to obtain a more complete view of complex traffic situations at intersections. 

Obtained object data from the employed perception sensors is sent to equipped vehicles within range of the intersection and can then be used for fusion with local perception data.


### 1.4 본 논문의 목적 

본 논문의 목적은 이웃차량이나 RSU를 이용하여 받은 정보를 퓨전하여 호스트 차량에서 물체 탐지 인지를 가능하게 하는 가상센서를 제안하다. `The goal of this paper is to present a virtual sensor approach in a generic high-level fusion architecture that processes incoming object data from communicating vehicles or roadside stations in a way so that its output is suitable for high-level fusion with object data from the local perception of the host vehicle. `

For this purpose, incoming object data is predicted to the current time step using a CTRA motion model in the objects’ spatially oriented body coordinate frame in combination with an unscented Kalman filter (UKF). 

This new approach allows for the incorporation of non-horizontal motion of communicating vehicles, for example on inclined road surfaces. 

In the second step, the predicted objects are transformed into the host vehicle’s local coordinate frame. 

In both preprocessing steps, a correct treatment of the state uncertainties is of great importance, because adequate uncertainty measures for the estimated states are vital for later fusion with the local perception’s output.

For the prediction, the unscented Kalman filter incorporates the process noise inherent to the prediction in the local coordinate frame directly into the global state of the received object. 

로컬 좌표로의 변형을 위해 두 방식이 비교 되었다. : **A linearized** &  **an unscented transformation** `For the transformation into the local coordinate frame, the consistency of two approaches, a linearized and an unscented transformation, are compared. `

성능 측정을 위해 **Kullback-Leibler divergence**가 사용되었다. `In this context, the Kullback-Leibler divergence is used as performance measure.`

The output of the virtual sensor is evaluated in a real world scenario concerning accuracy and consisteny using data from two experimental vehicles.

### 1.5 논문의 구성 

The rest of the paper is structured as follows: 
- Section II presents a high-level fusion architecture for cooperative perception. 
- In Section III, the concept of the virtual sensor for a Car2X-based perception is outlined. 
    - In this context, the two main steps of the virtual sensor approach, temporal and spatial alignment, are described. 
- Section IV provides experimental results with an implementation of the virtual sensor concept. 
- In Section V, conclusions are drawn.


## II. HIGH-LEVEL FUSION ARCHITECTURE FOR COOPERATIVE PERCEPTION

In automotive applications, different architectures for sensor data fusion have been studied in the past. In low-level
fusion architectures, raw data from the different sensors
is sent to a global fusion unit. Since sensor data is not
preprocessed before sending it to the fusion unit, a high data
bandwidth is required in this kind of architecture. Another
drawback of low-level fusion is its lacking modularity. Extending a low-level architecture with a new sensor requires
significant changes to the fusion module in general, since
raw data formats differ from sensor to sensor. In contrast to
that, high-level fusion architectures rely on the assumption
that every sensor preprocesses its raw data and provides the
central fusion unit with a local list of tracks, all including
the tracks’ states and covariances. Except for the fact that
the number of states estimated varies with each sensor,
the interface between the sensors and the central fusion
module is standardized. The central fusion module combines
the local track lists to a global one. For distributed sensor
networks, as in cooperative perception systems, a high-level
fusion architecture is preferable due to its reduced communication bandwidth and its high modularity. In [6], a fusion
architecture for cooperative perception systems is introduced.
This architecture is briefly described in the following.








