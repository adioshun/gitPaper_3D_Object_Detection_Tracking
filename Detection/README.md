# PointNet을 기반으로한 3D Point cloud 딥러닝 방법론 


> PointNet(2017) - PointNet2(2017) - Frustum-pointNet(2018) - RoarNet(2018)

이미지는 픽셀이라고 하는 2D 배열 형태의 표준화된 단일 데이터 구조체 및 표현 방법(Representive)을 가지고 있어 이를 기반으로한 여러 분석 방법론이 개발 되었습니다. 

하지만 3D Pointcloud 는 이러한 표준화된 표현 방법론(Representive)이 정의 되지 않아 많은 방식들이 제안되고 테스트 되고 있습니다. 

대표적인 표현 방법론은 
- Volumetric : 2D 픽셀을 3D 복셀로 확장 시켜 물체를 표현 하는 기법으로 
- Multiview 
- point cloud :

최근에는 Pointcloud 자체를 입력으로 받아 직접 학습을 하는 방식이 널리 사용되고 있다. 

먼저 이미지데이터와 달리 포인트 클라우드 데이터의 속성을 살펴 보면 다음과 같다. 
- Permutation Invariance : 데이터의 위치나 입력 순서에 영향을 받지 않아야 한다. 
- Interaction among points. (포인트간 상호성)
- Invariance under transformations (변화에 불변성) : 물체가 회전(rotating)하거나 위치(translating)가 변하더라도 결과에 영향을 미치지 않아야 한다. 
- Scales Invariance : 거리가 멀어 짐에 따라 포인트의 밀집도가 변함 


#### PointNet버젼별 도입된 주요 기능 
#####  v1
	- Symmetry Function : 입력 순서에 대한 강건성 확보 (?? Max Pool 레이어로 끝??)
    - Input/Freature Transform Net : 학습시 특정 공간으로 정렬하여 회전 등 변화에 대한 강건성 확보(=Pose Normalization)

##### v2
    - SA module : CNN에서는 일반화 성능을 올리기 위해 Local 특징이 중요, SA로 계층적 구조 형성 하여 가능  (구성 : Sample + Grouping + Mini Net)
    - SA module MSG(Multi-Scale Grouping) : 거리에 따른 밀집도 강건성 확보, 서로 다른 밀집도 끼리 특징 생성  
    - FP Module : SA로 축약화시 세그멘테이션수행을 위한 포인트별 특징 정보 사라짐, 서브셈플에서 원폰 포인트를 특징을 전파하여 문제 해결 
   
##### v3 
    - Center Regression Net
    - 3D Box Estimation Net 
    - Pointcloud Augmentation : 2D box augmentation, point cloud augmentation

---

## 1. Permutation Invariance

Symmetry Function을 이용하면 이러한 입력 순서에 대한 영향을 줄일수 있다. Symmetry Function은 입력으로 n개의 벡터를 받고 출력으로 입력 순서에 간건한 새 벡터를 출력 한다. +,x이 대표적인 Symmetry Function이다. 



> Sort와 RNN방식도 있다. 

---
## 2. Interaction among points.

이웃 포인트간의 관계를 보고 local structures를 구성 할수 있어야 하며, local structures간 연결 고리를 알수 있어야 한다. 




--- 

## 3. Invariance under transformations 

해결 방법은 특징 추출 전에 모든 입력을 정규공간(canonical space)에 정렬 하는것이다.

**T-net**을 이용하여 **아핀 매트릭스**를 획득 한후 입력 포인트클라우드에 바로 적용한다.



> T-Net은 deepMind의 STN을 기반으로 하였다. 

![](https://camo.githubusercontent.com/9a61e74fe903a0e0518376aa9a8607eb0963a95a/68747470733a2f2f692e696d6775722e636f6d2f4c5a69446631362e706e67)



## Scales Invariance 

**계층적 구조** 제안 : 중앙포인트를 기반으로 이웃 포인트(point-to-point relations)의 local region 관계를 학습 하므로  different scales에서의 local context학습이 가능하다. 

**abstraction levels** = Sampling layer, Grouping layer and PointNet layer
- Sampling layer : FPS알골리즘을 이용하여 local regions의 중앙점에 해당하는 포인트들 선정 
- Grouping layer : 중앙점을 기준으로 이웃점을 찾아 local region sets 구성 
- PointNet layer : mini-PointNet을 이용하여 local region sets 에서 Feature 벡터 추출 


abstraction levels을 추출된 각 scale의 Feature의 합치고 학습하기 위해 **Density adaptive PointNet layers** 제안



---

- [An In-Depth Look at PointNet](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a): PointNet_v1에 대한 요약 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NzE2ODAwNTUsMjUwODUyNTc0LDkxNz
Q3MjEzM119
-->