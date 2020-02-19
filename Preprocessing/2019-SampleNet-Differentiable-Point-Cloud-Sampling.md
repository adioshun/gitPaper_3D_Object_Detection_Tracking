# [SampleNet: Differentiable Point Cloud Sampling](https://arxiv.org/abs/1912.03663)


## 1. Introduction


A widely used method is farthest point sampling (FPS) [30, 52, 18, 27]. 

```
[30] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. Proceedings of Advances in Neural Information Processing Systems (NeuralIPS), 2017
[52] Lequan Yu, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, and Pheng Ann Heng. PU-Net: Point Cloud Upsampling Network. Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2790–2799, 2018.
[18] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. PointCNN: Convolution On XTransformed Points. Proceedings of Advances in Neural Information Processing Systems (NeuralIPS), 2018. 
[27] Charles R. Qi, Or Litany, Kaiming He, and Leonidas J.Guibas. Deep Hough Voting for 3D Object Detection in Point Clouds. Proceedings of the International Conference on Computer Vision (ICCV), 2019. 
```


FPS starts from a point in the set, and iteratively selects the farthest point from the points already selected [7,23]. 

```
[7] Yuval Eldar, Michael Lindenbaum, Moshe Porat, and Y. Yehoshua Zeevi. The Farthest Point Strategy for Progressive Image Sampling. IEEE Transactions on Image Processing, 6:1305–1315, 1997.
[23] Carsten Moenning and Neil A. Dodgson. Fast Marching farthest point sampling. Eurographics Poster Presentation, 2003.
```

It aims to achieve a maximal coverage of the input.

단점 : 기하학적 에러 최소화에 초점을 맞추어서 task(분류/세그멘테이션 등)의 이후 작업을 고려 하지 않고 있음 `FPS is task agnostic. It minimizes a geometric error and does not take into account the subsequent processing of the sampled point cloud. `

최근 연구에서 task를 고려한 샘플링 방법이 제안 되었다. `A recent work by Dovrat et al. [6] presented a task-specific sampling method. `
- Their key idea was to simplify and then sample the point cloud. 
- In the first step, they used a neural network to produce a small set of simplified points in the ambient space, optimized for
the task. 
- This set is not guaranteed to be a subset of the input. 
- Thus, in a post-processing step, they matched each simplified point to its nearest neighbor in the input point cloud, which yielded a subset of the input.





