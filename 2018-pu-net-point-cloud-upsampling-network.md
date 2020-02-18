# PU-Net: Point Cloud Upsampling Network


https://github.com/yulequan/PU-Net

https://arxiv.org/abs/1801.06761



## 1. Introduction


### 1.1 Related work: optimization-based methods. 

Alexa et al. [2] upsamples a point set by interpolating points at vertices of a Voronoi diagram in the local tangent space. 

Lipman et al. [24] present a locally optimal projection (LOP) operator for points resampling and surface reconstruction based on an L1 median. 
- The operator works well even when the input point set contains noise and outliers. 

Successively, Huang et al. [14] propose an improved weighted LOP to address the point set density problem.

위 방법들은 제약(가정)하에 작동 한다. `Although these works have demonstrated good results, they make a strong assumption that`
- the underlying surface is smooth, thus restricting the method’s scope. 


##### 제약 해결 방안 
Then, Huang et al. [15] introduce an edge-aware point set resampling method by first resampling away from edges and then progressively approaching edges and corners. 
- However, the quality of their results heavily relies on the accuracy of the normals at given points and careful parameter tuning. 

It is worth mentioning that Wu et al. [35] propose a deep points representation method to fuse consolidation and completion in one coherent step. 
- Since its main focus is on filling large holes, global smoothness is, however, not enforced, so the method is sensitive to large noise. 

위 방법들은 데이터 기반이 아니기에 사전정보에 의존적이다. `Overall, the above methods are not data-driven, thus heavily relying on priors.`



### 1.2 Related work: deep-learning-based methods. 

Points in a point cloud do not have any specific order nor follow any regular grid structure, so only a few recent works adopt a deep learning model to directly process point clouds. 

Most existing works convert a point cloud into some other 3D representations such as the volumetric grids [27, 36, 31, 6] and geometric graphs [3, 26] for processing. 

Qi et al. [29, 30] firstly introduced a deep learning network for point cloud classification and segmentation; in particular, the PointNet++ uses a hierarchical feature learning architecture to capture both local and global geometry context. 

Subsequently, many other networks were proposed for high-level analysis problems with point clouds [18, 13, 21, 34, 28].

However, they all focus on global or mid-level attributes of point clouds. 

In another work, Guerrero et al. [10] developed a network to estimate the local shape properties in point clouds, including normal and curvature. 

Other relevant networks focus on 3D reconstruction from 2D images [8, 23, 9]. To the best of our knowledge, there are no prior works focusing on point cloud upsampling.

## 2. Network Architecture

