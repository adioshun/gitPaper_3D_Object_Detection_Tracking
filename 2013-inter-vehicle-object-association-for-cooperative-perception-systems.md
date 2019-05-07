# [Inter-Vehicle Object Association for Cooperative Perception Systems](https://ieeexplore.ieee.org/abstract/document/6728345)





협조 인지 시스템이란? In cooperative perception systems, different vehicles share object data obtained by their local environment perception sensors, like radar or lidar, via wireless communication. Inaccurate self-localizations of the vehicles complicate association of locally perceived objects and objects detected and transmitted by other vehicles. 

제안 내용 In this paper, a method for intervehicle object association is presented. 
- Position and orientation offsets between object lists from different vehicles are estimated by applying point matching algorithms. 
- Different algorithms are analyzed in simulations concerning their robustness and performance. 

결론 : Results with a first implementation of the socalled Auction-ICP algorithm in a real test vehicle validate the simulation results.

## I. INTRODUCTION


Current research projects like simTD [1] try to exploit the benefits of wireless communication for advanced driver assistance systems. Therefor, every equipped vehicle and roadside station broadcasts its own position and dynamic state. 
- With this information, assistance systems like traffic light assistance, local hazard warning or cross traffic assistance can be realized. 

The project initiative Ko-FAS [2] with its joint project Ko-PER tries to enhance the scope of communication-based assistance systems by providing driver assistance systems a global view of the traffic environment.
- For this reason, every equipped vehicle and roadside station not only broadcasts its own position and state, but also a model of its dynamic environment based on its local perception sensors. 
- Thus, each equipped vehicle and roadside station can help to enhance the field of view of the other ones.
- This technology, called cooperative perception, will empower new forms of driver assistance and safety systems.



### 1.1 문제점 

A major challenge for cooperative perception systems and communication-based assistance systems in general is the **association of data** from different sources. 

In a cooperative perception system, a vehicle receives information about dynamic objects from other equipped vehicles or roadside stations. 

In order to combine these perception data with its own local environment model, temporal and spatial alignment is crucial. 

### 1.2 기존 해결 책 

In [3], a method for temporal and spatial alignment in a cooperative perception system is proposed. 

However, the results of spatial alignment very much depend on the quality of the communication partners’ self-localizations. 

고정 물체의 경우 컨트롤된 상황에서 GPS이용 하여 측정하면 된다. `For static entities like roadside stations, the position and orientation can be determined by highly accurate means like carrier-phase differential GPS systems under optimal conditions.`

그러나 차량의 경우는 GPS가 효율적이지 않다. `However, vehicles often suffer from localization errors caused for example by GPS outage or multi-path effects in urban environments.`

As soon as the self-localizations of the vehicles become deficient, the relative position and orientation of transmitted objects in the host vehicle’s reference frame become inaccurate, too. 

Such errors can have a strong negative impact on the quality of object association and fusion. 

Furthermore, there is no way to correct the state of objects which are only seen by the sender. 

### 1.3 본 논문의 목적 

The goal of this paper is to reduce the negative impact of inaccurate self-localizations on the association and fusion quality. 

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


















