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

---

## 1. Permutation Invariance

Symmetry Function을 이용하면 이러한 입력 순서에 대한 영향을 줄일수 있다. Symmetry Function은 입력으로 n개의 벡터를 받고 출력으로 입력 순서에 간건한 새 벡터를 출력 한다. +,x이 대표적인 Symmetry Function이다. 


```python 
# Symmetric function: max pooling
net = tf_util.max_pool2d(net, [num_point,1],padding='VALID', scope='maxpool')
net = tf.reshape(net, [batch_size, -1])
net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,scope='fc1', bn_decay=bn_decay)
net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope='dp1')
net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,scope='fc2', bn_decay=bn_decay)
net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope='dp2')
net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
```


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


```python 

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform

def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform

```

## Scales Invariance 

**계층적 구조** 제안 : 중앙포인트를 기반으로 이웃 포인트(point-to-point relations)의 local region 관계를 학습 하므로  different scales에서의 local context학습이 가능하다. 

**abstraction levels** = Sampling layer, Grouping layer and PointNet layer
- Sampling layer : FPS알골리즘을 이용하여 local regions의 중앙점에 해당하는 포인트들 선정 
- Grouping layer : 중앙점을 기준으로 이웃점을 찾아 local region sets 구성 
- PointNet layer : mini-PointNet을 이용하여 local region sets 에서 Feature 벡터 추출 


abstraction levels을 추출된 각 scale의 Feature의 합치고 학습하기 위해 **Density adaptive PointNet layers** 제안




