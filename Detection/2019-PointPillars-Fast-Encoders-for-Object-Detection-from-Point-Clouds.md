# [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

https://researchcode.com/code/3089170440/pointpillars-fast-encoders-for-object-detection-from-point-clouds/

> CVPR2019, SECOND기반 



## [영문정리](https://medium.com/@m7807031/pointpillars-fast-encoders-for-object-detection-from-point-clouds-brief-3c868c5c463d)

![](https://i.imgur.com/BzJHmN3.png)

구성 요소 3가지 `It can be seen that PP is mainly composed of three major parts:  `
1. Pillar Feature Net -> 2. Backbone (2D CNN) -> 3. Detection Head (SSD)

**1. Pillar Feature Net**  
- Pillar Feature Net will first scan all the point clouds with the **overhead view** and build the pillars per unit of xy grid. 
	- It is also the basic unit of the feature vector of PP. 
- By calculating the point cloud in each Pillar, you can get D = for each point cloud. [x, y, z, r, xc, yc, zc, xp, yp]  among them:  
	- x,y,z,r is a single cloud x, y, z, reflection  
	- Xc, yc, zc is the point cloud point from the geometric center point of the pillar  
	- Xp, yp is the distance from the center of pillar x, y

Then combine the information into [D, P, N] superimposed tensors  among them:  
- D is the point cloud D  
- P is Pillar’s index  
- N is the point cloud index of the Pillar

Next,  
- Use a simple version of pointNet (a linear layer containing Batch-Norm and ReLu) to generate (C, P, N) tensors, 
- then generate (C, P) tensors, 
- and finally produce (C, H, W) Point cloud feature image (2D).

**2. Backbone**  
Can refer to the picture for calculation

**3. Detection Head**  
PP uses Single Shot Detector for bounding-box calculations

---

## [영문정리2](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/point_pillars.html)

라이다 데이터를 **필라**형태로 묶고, 포인트넷으로 인코딩 하여 2D BEV pseudo-이미지 생성. `Group lidar data into pillars and encode them with pointnet to form a 2D birds view pseudo-image.` 2D 이미지 탐지 네트워크인 SSD를 이용하여 물체 탐지 수행 `SSD with 2D convolution is used to process this pseudo image for object detection. `It achivees the SOTA at 115 Hz.

#### Overall impression

This paper follows the line of work of [VoxelNet](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/voxelnet.md) and [SECOND](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/second.md) and improves the encoding methods. 

Both voxelnet and SECOND encode point cloud into **3D voxels** and uses expensive 3D convolution. The main contribution of this paper lies in that it encodes (“sacrifices”) the information of the relatively unimportant dimension of z into different channels of the 2D pseudo image. This greatly boosts the inference.

#### Key ideas[](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/point_pillars.html#key-ideas)

-  복셀대신 **필라**상에 동작 한다.  PointPillars operates on pillars instead of voxels and eliminates the need to tune binning of the vertical direction by hand. All operation on the pillars are 2D conv, which can be highly optimized on GPU.

-   The sparsity of point cloud bird’s eye view is exploited by fixed number of non-empty pillars per sample (P=12000) and number of points per pillar (N=100). This creates a tensor of (D, P, N). D=9 is the dimension of the augmented (“decorated”) lidar point (x, y, z, reflectance r, center of mass of all points in the pillar xc, yc, zc, offset form xc and yc, xp and yp).

-   Pillar encoding (D, P, N) –> (C, P, N) –> (C, P) –> (C, H, W). The last step is to scatter the pillars back to the 2D image.

-   Detection head uses SSD with 2D anchor boxes. Bounding box height and elevation in z direction were not used for anchor box matching but rather used as additional regression target.

-   Loss is standard object detection loss (with additional parameters h, l and heading angle $\theta$. The encoding of heading angle is $\sin\theta$ and cannot differentiate opposite directions with difference of $\pi$. Additional heading classification is used.

-   Adjusting the size of spatial binning will impact the speed (and accuracy).

-   **Point cloud encoding method**:
    -   MV3D uses M+2 channel BV images
    -   Complex YOLO uses 1+2, largely follows MV3D
    -   PIXOR pixelates into 3D voxels of occupancy (0 or 1), but treats z as diff channels of 2D input
    -   VoxelNet pixelates into 3D voxels, but the pixelation process is by PointNet
    -   PointPillars pixelates into 2D voxels, but the pixelation process is by PointNet

#### Technical details[](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/point_pillars.html#technical-details)

-   A lidar robotics pipeline uses a bottom up approach involving BG subtraction, spatiotemporal clustering and classification.

-   A common way to process point cloud is to project it to regular 2D grid from bird’s eye view.  [MV3D](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/mv3d.html)  and  [AVOD](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/avod.html)  relies on fixed feature extractors. VoxelNet is the first end-to-end method to learn features in each voxel bin of a point cloud. SECOND improves upon VoxelCloud and speeds up the inference.
-   The point cloud is a sparse representation and an image is dense. Bird’s eye view is extremely sparse, but also creates opportunities for extreme speedup.
-   Lidar point cloud not in the FOV of camera is discarded.
-   Lidars are usually mounted on top of a car and the mass center of z is ~0.6 m.
-   TensorRT gives about 50% speedup from PyTorch.
-   KITTI lidar image is 40m x 80m.
-   Adding the off-center offset xp and yp would actually boost AP. So yes,  **representation matters**.

#### Notes[](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/point_pillars.html#notes)

-   This work can be applied to radar and multiple lidar data.
-   The encoding of heading angle can be improved by using both $\sin\theta$ and $\cos\theta$, as in AVOD.
-   The  [code](https://github.com/nutonomy/second.pytorch)  is based on the code base of SECOND.