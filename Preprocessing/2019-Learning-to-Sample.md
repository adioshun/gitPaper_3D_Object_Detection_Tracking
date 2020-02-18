# Learning to Sample

https://github.com/orendv/learning_to_sample

https://arxiv.org/abs/1812.01659

## 2. Related work

### 2.1 Point cloud simplification and sampling 

Several techniques for either point cloud simplification [31, 32] or sampling [33, 34] have been proposed in the literature. 

```
[31] M. Pauly, M. Gross, and L. P. Kobbelt, “Efficient Simplification of Point-Sampled Surfaces,” Proceedings of IEEE Visualization Conference, 2002. 2
[32] C. Moenning and N. A. Dodgson, “A new point cloud simplification algorithm,” Proceedings of the IASTED International Conference on Visualization, Imaging and Image Processing (VIIP), 2003. 2
[33] S. Katz and A. Tal, “Improving the Visual Comprehension of Point Sets,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 121–128,2013. 2
[34] S. Chen, D. Tian, C. Feng, A. Vetro, and J. Kovaevi,“Fast Resampling of Three-Dimensional Point Clouds via Graphs,” IEEE Transactions on Signal Processing, vol. 66, pp. 666–681, 2018. 2
```

#### A. simplification

Pauly et al. [31] presented and analyzed several simplification methods for point-sampled surfaces, including: clustering methods, iterative simplification and particle simulation. 

The simplified point set, resulting from these algorithms, was not restricted to be a subset of the original one. 

Farthest point sampling was adopted in the work of Moenning and Dodgson [32] as a means to simplify point clouds of geometric shapes, in a uniform as well as feature-sensitive manner.

#### B. sampling

Katz and Tal [33] suggested a view dependent algorithm to reduce the number of points. 

They used hidden-point removal and target-point occlusion operators in order to improve a human comprehension of the sampled point set. 


Recently, Chen et al. [34] employed graph-based filters to extract per point features. 

Points that preserve specific information are likely to be selected by a their sampling strategy. 

The desired information is assumed to be beneficial to a subsequent application.

The above sampling approaches aim to optimize a variety of sampling objectives. 

However, they do not consider directly the objective of the task to be followed.


### 2.2 Progressive simplification 

In a seminal paper, Hoppe [35] proposed a technique for progressive mesh simplification. 

In each step of his method, one edge is collapsed such that minimal geometric distortion is introduced.

```
[35] H. Hoppe, “Progressive Meshes,” Proceedings of the ACM Special Interest Group on Computer Graphics (SIGGRAPH), pp. 99–108, 1996
```

A recent work by Hanocka et al. [36] suggested a neural network that performs task-driven mesh simplification. 

Their network relies on the edges between the mesh vertices. This information is not available for point clouds. 

```
[36] R. Hanocka, A. Hertz, N. Fish, R. Giryes, S. Fleishman, and D. Cohen-Or, “MeshCNN: A Network with an Edge,” arXiv preprint arXiv:1809.05910, 2018.
```

Several researchers studied the topic of point set compression [37, 38, 39]. 

An octree data structure was used for progressive encoding of the point cloud. 

The objective of the compression process was low distortion error.
```
[37] J. Peng and C.-C. J. Kuo, “Octree-Based Progressive Geometry Encoder,” Proceedings of SPIE, pp. 301–311, 2003. 2 
[38] Y. Huang, J. Peng, C.-C. J. Kuo, and M. Gopi, “Octree-Based Progressive Geometry Coding of Point Clouds,” Proceedings of the Eurographics Symposium on Point-Based Graphics,pp. 103–110, 2006. 2
[39] R. Schnabel and R. Klein, “Octree-based Point-Cloud Compression,” Proceedings of the Eurographics Symposium on Point-Based Graphic, 2006. 2
```

### 2.3 Deep learning on point sets 

[PointNet1] The pioneering work of Qi et al. [1] presented PointNet, the first neural network that operates directly on unordered point cloud data. 
- They constructed their network from per-point multi-layer perceptrons, a symmetric pooling operation and several fully connected layers. 
- PointNet was employed for classification and segmentation tasks and showed impressive results. 
- PointNET에서는 입력 데이터수를 줄이기 위해 **random sampling** and **FPS**를 사용하였다. `For assessing the applicability of PointNet for reduced number of input points, they used random sampling and FPS. `

본 논문의 제안 `In our work,` we suggest a data-driven sampling approach, that improves the classification performance with sampled sets in comparison to these sampling methods.

[PointNet2] Later on, Qi et al. extended their network architecture for hierarchical feature learning [2]. 
- In the training phase, centroid points for local feature aggregation were selected by FPS. 
- Similar to their previous work [1], FPS was used for evaluating the ability of their network to operate on fewer input points.

Li et al. [11] suggested to learn the centroid points for feature aggregation by a self-organizing map (SOM). 
- They used feature propagation between points and SOM nodes and showed improved results for point cloud classification and part segmentation. 
- The SOM was optimized separately, as a pre-processing step.

Building on the work of Qi et al. [1], Achlioptas et al. [17] developed autoencoders and generative adversarial networks for point clouds. 
- Instead of shape class or perpoint label, the output of their network was a set of 3D points. 
- In this work we apply our sampling approach for the task of point cloud reconstruction with the autoencoder proposed by Achlioptas et al.

Several researchers tackled the problem of point cloud consolidation. 

Yu et al. [24] extracted point clouds from geodesic patches of mesh models. 
- They randomly sampled the point sets and trained a network to reconstruct the original patch points. 
- Their follow-up work [25] incorporated edge information to improve the reconstruction accuracy.

Zhang et al. [26] studied the influence of sampling strategy on point cloud up-sampling. 
- They used Monte-Carlo random sampling and curvature based sampling. 
- Their network was trained to produce a fixed size point cloud from its sample. 

In contrast to these works, our study focuses on the down-sampling strategy.