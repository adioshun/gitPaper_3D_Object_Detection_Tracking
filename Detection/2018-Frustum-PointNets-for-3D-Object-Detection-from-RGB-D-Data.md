
# [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf)

> [홈페이지](http://stanford.edu/~rqi/frustum-pointnets/), [깃허브](https://github.com/charlesq34/frustum-pointnets), [Article](https://blog.csdn.net/shuqiaos/article/details/82752100)

In this work, we study 3D object detection from RGBD data in both indoor and outdoor scenes. While previous methods focus on images or 3D voxels, often obscuring natural 3D patterns and invariances of 3D data, we directly operate on raw point clouds by popping up RGB-D scans. However, a key challenge of this approach is how to efficiently localize objects in point clouds of large-scale scenes (region proposal). Instead of solely relying on 3D proposals, our method leverages both mature 2D object detectors and advanced 3D deep learning for object localization, achieving efficiency as well as high recall for even small objects. Benefited from learning directly in raw point clouds, our method is also able to precisely estimate 3D bounding boxes even under strong occlusion or with very sparse points. Evaluated on KITTI and SUN RGB-D 3D detection benchmarks, our method outperforms the state of the art by remarkable margins while having real-time capability.



## 1. Introduction

본 논문에서는 3D 인지 시스템에 대하여 다루고 있다. (탐지, 분류, 방향) `Recently, great progress has been made on 2D image understanding tasks, such as object detection [13] and instance segmentation [14]. However, beyond getting 2D bounding boxes or pixel masks, 3D understanding is eagerly in demand in many applications such as autonomous driving and augmented reality (AR). With the popularity of 3D sensors deployed on mobile devices and autonomous vehicles, more and more 3D data is captured and processed. In this work, we study one of the most important 3D perception tasks – 3D object detection, which classifies the object category and estimates oriented 3D bounding boxes of physical objects from 3D sensor data.`

3D 센서 데이터가 포인트 클라우드 형태로 저장 되지만 이를 분석 하는 기술은 아직 연구 분야 이다. 기존에는 포인트 클라우드를 **이미지로 투영**하거나 **volumetric grids by quantization**로 만들어 CNN에 적용 하였다. `While 3D sensor data is often in the form of point clouds, how to represent point cloud and what deep net architectures to use for 3D object detection remains an open problem. Most existing works convert 3D point clouds to images by projection [36, 26] or to volumetric grids by quantization [40, 23, 26] and then apply convolutional networks.`

**PointNets**에서는 위와 같은 변경 과정 없이 포인트 클라우드를 바로 사용하여 좋은 성과를 보였다. `This data representation transformation, however, may obscure natural 3D patterns and invariances of the data. Recently, a number of papers have proposed to process point clouds directly without converting them to other formats. For example, [25, 27] proposed new types of deep net architectures, called PointNets, which have shown superior performance and efficiency in several 3D understanding tasks such as object classification and semantic segmentation.`

포인트넷이 분류 기능은 좋지만 탐색은 별도의 문제이다. 이를 위해서 후보 영역 추천 하는 기능이 있어야한다. 2D의 기술을 활용하여 슬라이딩 윈도우 기법을 쓰거나 3D Region Proposal Network를 사용할수도 있다. 하지만 이들은 컴퓨팅 부하가 크다. `While PointNets are capable of classifying a whole point cloud or predicting a semantic class for each point in a point cloud, it is unclear how this architecture can be used for instance-level 3D object detection. Towards this goal, we have to address one key challenge: how to efficiently propose possible locations of 3D objects in a 3D space. Imitating the practice in image detection, it is straightforward to enumerate candidate 3D boxes by sliding windows [8] or by 3D region proposal networks such as [33]. However, the computational complexity of 3D search typically grows cubically with respect to resolution and becomes too expensive for large scenes or real-time applications such as autonomous driving.`

![](https://i.imgur.com/r5ipOrn.png)

본 논문에서는 차원 축소를 통해서 탐색 범위를 줄였다. 2D 이미지 기반 탐지기의 기능을 활용 하였다. `Instead, in this work, we reduce the search space following the dimension reduction principle: we take the advantage of mature 2D object detectors (Fig. 1).`
- First, we extract the 3D bounding frustum of an object by extruding 2D bounding boxes from image detectors. 
- Then, within the 3D space trimmed by each of the 3D frustums, we consecutively perform **3D object instance segmentation** and **amodal 3D bounding box regression** using two variants of **PointNet**. 
	- The segmentation network predicts the 3D mask of the object of interest (i.e. instance segmentation); 
	- and the regression network estimates the amodal 3D bounding box (covering the entire object even if only part of it is visible).

이전방식에서도 RGB-D의 2D map을 CNN에 적용하는 방식이 있었지만 본 논문의 제안에서는 좀더 **3D-centric** 하다. In contrast to previous work that treats RGB-D data as 2D maps for CNNs, our method is more 3D-centric as we lift depth maps to 3D point clouds and process them using 3D tools. This 3D-centric view enables new capabilities for exploring 3D data in a more effective manner. 
- First, in our pipeline, a few transformations are applied successively on 3D coordinates, which align point clouds into a sequence of more constrained and canonical frames. These alignments factor out pose variations in data, and thus make 3D geometry pattern more evident, leading to an easier job of 3D learners. 
- Second, learning in 3D space can better exploits the geometric and topological structure of 3D space. In principle, all objects live in 3D space; therefore, we believe that many geometric structures, such as repetition, planarity, and symmetry, are more naturally parameterized and captured by learners that directly operate in 3D space. The usefulness of this 3D-centric network design philosophy has been supported by much recent experimental evidence.

Our method achieve leading positions on KITTI 3D object detection [1] and bird’s eye view detection [2] benchmarks. Compared with the previous state of the art [6], our method is 8.04% better on 3D car AP with high efficiency (running at 5 fps). Our method also fits well to indoor RGBD data where we have achieved 8.9% and 6.4% better 3D mAP than [16] and [30] on SUN-RGBD while running one to three orders of magnitude faster


---

https://medium.com/@yckim/%EC%A0%95%EB%A6%AC-roarnet-a-robust-3d-object-detection-based-on-region-approximation-refinement-91c66201eaf2
