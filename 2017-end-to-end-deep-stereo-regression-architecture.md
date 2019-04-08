[End-to-end deep stereo regression architecture](http://openaccess.thecvf.com/content_ICCV_2017/papers/Kendall_End-To-End_Learning_of_ICCV_2017_paper.pdf)

https://arxiv.org/abs/1703.04309

코드 : GC-Net, [pythorch](https://github.com/zyf12389/GC-Net), [TF](https://github.com/Jiankai-Sun/GC-Net), [TF](https://github.com/MaidouPP/gc_net_stereo)

```
- A deep learning architecture for regressing disparity from a rectified pair of stereo images.
- Leverage knowledge of the problem’s geometry to form a cost volume using deep feature representations.
- Learn to incorporate contextual information using 3-D convolutions over this volume.
- Disparity values regressed from the cost volume using a differentiable soft argmin operation, which allows to train end-to-end to sub-pixel accuracy without any additional post-processing or regularization.
```


참고 : 2018, Pyramid Stereo Matching Network, [[code]](https://github.com/JiaRenChang/PSMNet)