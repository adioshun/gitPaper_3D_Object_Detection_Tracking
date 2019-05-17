# LIDAR-based 3D Object Perception

https://velodynelidar.com/lidar/hdlpressroom/pdf/Articles/LIDAR-based%203D%20Object%20Perception.pdf

라이다 기반 탐지, 분류, 추적을 다룬다. `This paper describes a LIDAR-based perception system for ground robot mobility, consisting of 3D object detection, classification and tracking. `

자동차에 적용 했다. `The presented system was demonstrated on-board our autonomous ground vehicle MuCAR-3, enabling it to safely navigate in urban traffic-like scenarios as well as in off-road convoy scenarios. `

2D, 3D 데이터 처리 기술을 병행 적용 하였다. `The efficiency of our approach stems from the unique combination of 2D and 3D data processing techniques. `
- Whereas fast segmentation of point clouds into objects is done in a **2 1/2D** occupancy grid, 
- classifying the objects is done on raw **3D** point clouds. 

For fast switching of domains, the occupancy grid is enhanced to act like a hash table for retrieval of 3D points. 

In contrast to most existing work on 3D point cloud classification, where realtime operation is often impossible, this combination allows our system to perform in real-time at 0.1s frame-rate.


## I. INTRODUCTION