|논문명|Survey on Ranging Sensors and Cooperative Techniques for Relative Positioning of Vehicles|
|-|-|
|저자(소속)|Fabian De Ponte Müller (Institute of Communications and Navigation)|
|학회/년도| Sensors 2017, [논문](https://www.mdpi.com/1424-8220/17/2/271)|
|키워드||
|참고||
|코드||


# Survey on Ranging Sensors and Cooperative Techniques for Relative Positioning of Vehicles

Future driver assistance systems will rely on accurate, reliable and continuous knowledge on the position of other road participants, including pedestrians, bicycles and other vehicles.

The usual approach to tackle this requirement is to use on-board ranging sensors inside the vehicle.

Radar, laser scanners or vision-based systems are able to detect objects in their line-of-sight. 

In contrast to these non-cooperative ranging sensors, cooperative approaches follow a strategy in which other road participants actively support the estimation of the relative position. 

The limitations of on-board ranging sensors regarding their detection range and angle of view and the facility of blockage can be approached by using a cooperative approach based on vehicle-to-vehicle communication. 

The fusion of both, cooperative and non-cooperative strategies, seems to offer the largest benefits regarding accuracy, availability and robustness. 

This survey offers the reader a comprehensive review on different techniques for vehicle relative positioning. 

The reader will learn the important performance indicators when it comes to relative positioning of vehicles, the different technologies that are both commercially available and currently under research, their expected performance and their intrinsic limitations. 

Moreover, the latest research in the area of vision-based systems for vehicle detection, as well as the latest work on GNSS-based vehicle localization and vehicular communication for relative positioning of vehicles, are reviewed. 

The survey also includes the research work on the fusion of cooperative and non-cooperative approaches to increase the reliability and the availability.

## 1. Introduction

ADAS에서 자차와 이웃차량의 상대위치 속도 정보는 중요하다. `Advanced driver assistance systems play an important role in increasing the safety and efficiency of today’s roads, while the knowledge about the position of other vehicles is a fundamental prerequisite for numerous safety-critical applications in the Intelligent Transportation System (ITS) domain. Safety-critical applications, as for instance Forward Collision Avoidance (FCA), Lane Change Assistance (LCA) or Automatic Cruise Control (ACC), need continuous knowledge about the relative position and relative velocity of other vehicles in the vicinity of the ego vehicle.`

많은 센서의 발전으로 인지 성능이 발전 하였다. `For almost a decade, relative positioning sensors, such as radar sensors, have been available in commercial vehicles. In the last few years, camera systems have found their way into high-end vehicles for collision avoidance, lane-keeping assistance and in-vehicle traffic sign recognition. The first prototypes of fully-autonomous vehicles use 3D laser scanners to obtain an accurate representation of the surrounding environment. The richness and high precision of these devices makes it possible for autonomous vehicles to obtain a detailed representation of the scenery including the exact position of buildings, vegetation, other road participants and further obstacles. In this way, the robotic vehicle is able to self-localize itself and navigate through traffic [1].`

자동차 통신의 발전으로 이웃차량의 주변정보를 수신함으로써 Range 센서의 탐지 범위가 확대 되었다. V2V를 위한 메시지들이 정의 되었으며 각 메시지는 위치, 속도, 방향등의 정보를 브로드 캐스트 한다. 포함된 위치 정보를 이용하여서 이웃차량의 위치를 알수 있다. 이정보를 이용하여 센서로 탐지된 정보를 보정 할수 있다.   `With the standardization of the first Vehicle-to-Vehicle (V2V) communication protocols in Europe, America and Japan, cooperative approaches have made it possible to extend the perception range of the ego vehicle beyond the capabilities of on-board ranging sensors by using information from other vehicles in the surroundings. The European Telecommunications Standards Institute (ETSI) and the U.S. Society of Automotive Engineers (SAE) are currently working on the definition of different safety-critical messages for the V2V technology. Each vehicle will transmit periodically Cooperative Awareness Messages (CAMs) [2] or Basic Safety Messages (BSMs) [3] containing basic information, such as position, speed and heading.  The included position in global coordinates can be used by a vehicle to estimate its neighbors’ positions. Its own coordinates might be estimated using a Global Navigation Satellite System (GNSS), like the American Global Positioning System (GPS) or the European Galileo system. This estimate can be additionally enhanced by supporting it with on-board sensors, such as wheel angle, odometer and inertial sensors.`

Although it is demonstrated that autonomous road vehicles can rely solely on their on-board perception sensors, it is foreseen that they will greatly profit from the introduction of an inter-vehicle communication. Besides an increased availability and reliability in cooperative relative positioning, the communication enables cooperative perception by sharing sensor information and the execution of collaborative maneuvers between automated road vehicles. In this way, a higher degree of safety is achievable without sacrificing efficiency by driving with large safety distances and increased caution.

