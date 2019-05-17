# Vehicle Detection and Pose Estimation in Autonomous Convoys

https://uis.brage.unit.no/uis-xmlui/handle/11250/2455922




## 1. Introduction 


## 2. Background 

### 2.1 Previous work

기존 프로젝트들 


### 2.2 POS/POSIT

앞자의 방향 및 위치는 예측 되어야 한다. `The leading vehicle’s orientation and position relative the ego-vehicle is to be estimated.`

물체의 3D model이 알려져 있다는 가정 하에 **POS/POSIT방법**을 이용하면 **이미지** 한장으로도 예측이 가능하다. `Assuming a 3D model of the object is known, i.e. the relative geometry of a set of feature points, the translation and rotation matrices can be approximated by means of the POS/POSIT method from a single image.`


The methods POS(`Pose from Orthography and Scaling`), and POSIT(`POS with Iterations`) was first presented by DeMenthon et al. in [21]. 

요구 사항 `The methods require `
- minimum four known pairs of 3D feature point coordinates and 
- the corresponding 2D image coordinates.

Based on perspective projection the POS method generates a linear equation system based on the given feature point coordinate pairs. 

When solved an approximate of the current rotation and translation matrices of that object in relation to the camera position is estimated. [21] [22]

POSIT is an extended version of POS. 

By implementing POS in an iteration loop the results can be used to estimate an even more accurate approximation. 

The number of iteration can be specified according to available processing time. 

POSIT converges after only a few iterations, see section 16 in [21].

### 2.3 Imaging geometry


### 2.4 RANSAC

---

## 3. Methods

### 3.1 Vehicle Detection 


#### A.Vehicle Properties 

##### 가. Visual Properties 

colour, shape and illumination

##### 나. Geometrical properties








