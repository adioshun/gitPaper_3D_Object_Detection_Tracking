# [Multiple Object Tracking: A Literature Review](https://arxiv.org/pdf/1409.7618.pdf)



MOT는 오랜 기간 연구되었지만 아직 부족하다. 본 논문을 통해 리뷰를 하려 한다. `Multiple Object Tracking (MOT) is an important computer vision problem which has gained increasing attention due to its academic and commercial potential. Although different kinds of approaches have been proposed to tackle this problem, it still remains challenging due to factors like abrupt appearance changes and severe object occlusions. In this work, we contribute the first comprehensive and most recent review on this problem. We inspect the recent advances in various aspects and propose some interesting directions for future research. To the best of our knowledge, there has not been any extensive review on this topic in the community. We endeavor to provide a thorough review on the development of this problem in recent decades.` 

The main contributions of this review are fourfold: 
- 1) Key aspects in a multiple object tracking system, including formulation, categorization, key principles, evaluation of an MOT are discussed. 
- 2) Instead of enumerating individual works, we discuss existing approaches according to various aspects, in each of which methods are divided into different groups and each group is discussed in detail for the principles, advances
and drawbacks. 
- 3) We examine experiments of existing publications and summarize results on popular datasets to provide quantitative comparisons. We also point to some interesting discoveries by analyzing these results. 
- 4) We provide a discussion about issues of MOT research, as well as some interesting directions which could possibly become potential research effort in the future.


## 1 INTRODUCTION

MOT에서 대상은 사람, 차량, 운동선수등이 될수 있다. `Multiple Object Tracking (MOT), or Multiple Target Tracking (MTT), plays an important role in computer vision. The task of MOT is largely partitioned to locating multiple objects, maintaining their identities, and yielding their individual trajectories given an input video. Objects to track can be, for example, pedestrians on the street [1], [2],vehicles in the road [3], [4], sport players on the court [5], [6], [7], or groups of animals (birds [8], bats [9], ants [10],fish [11], [12], [13], cells [14], [15], etc.). Multiple “objects” could also be viewed as different parts of a single object[16]. `

본 논문에서는 보행자 추적에 중점을 두었다. 그 이유는 아래와 같다. `In this review, we mainly focus on the research on pedestrian tracking. The underlying reasons for this specification are threefold.`
- First, compared to other common objects in our environment, pedestrians are typical nonrigid objects, which is an ideal example to study the MOT problem. 
- Second, videos of pedestrians arise in a huge number of practical applications, which further results in great commercial potential. 
- Third, according to all data collected for this review, at least 70% of current MOT research efforts are devoted to pedestrians.


As a mid-level task in computer vision, multiple object tracking grounds high-level tasks such as pose estimation [17], action recognition [18], and behavior analysis [19]. 

It has numerous practical applications, such as visual surveillance [20], human computer interaction [21] and virtual reality [22]. These practical requirements have sparked enormous interest in this topic. 

Compared with Single Object Tracking (SOT), which primarily focuses on designing sophisticated appearance models and/or motion models to deal with challenging factors such as scale changes, out of-plane rotations and illumination variations, multiple object tracking additionally requires two tasks to be solved:
- determining the number of objects, which typically varies over time, 
- and maintaining their identities. 

Apart from the common challenges in both SOT and MOT, further key issues that complicate MOT include among others:
- 1) frequent occlusions, 
- 2) initialization and termination of tracks, 
- 3) similar appearance, and 
- 4) interactions among multiple objects. 

In order to deal with all these issues, a wide range of solutions have been proposed in the past decades. These solutions concentrate on different aspects of an MOT system, making it difficult for MOT researchers, especially newcomers, to gain a comprehensive understanding of this problem. Therefore, in this work we provide a review to discuss the various aspects of the multiple object tracking problem.

### 1.1 Differences from Other Related Reviews

다른 리뷰 논문들 요약 

### 1.2 Contributions 

### 1.3 Organization of This Review

### 1.4 Denotations

---

## 2 MOT PROBLEM

### 2.1 Problem Formulation

### 2.2 MOT Categorization

#### 2.2.1 Initialization Method

Most existing MOT works can be grouped into two sets [49], depending on how objects are initialized: 
- Detection-Based Tracking (DBT) and 
- Detection-Free Tracking (DFT).

##### Detection-Based Tracking (=tracking-by-detection)

As shown in Figure 1 (top), objects are first detected and then linked into trajectories. This strategy is also commonly referred to as “tracking-by-detection”. 

Given a sequence, 
- type-specific object detection or motion detection (based on background modeling) [50], [51] is applied in each frame to obtain object hypotheses,
- then (sequential or batch) tracking is conducted to link detection hypotheses into trajectories. 

There are two issues worth noting. 
- First, since the object detector is trained in advance, the majority of DBT focuses on specific kinds of targets, such as pedestrians, vehicles or faces. 
- Second, the performance of DBT highly depends on the performance of the employed object detector


##### Detection-Free Tracking

As shown in Figure 1 (bottom), DFT [52], [53], [54], [55] requires manual initialization of a fixed number of objects in the first frame, then localizes these objects in subsequent frames.

DBT is more popular because new objects are discovered and disappearing objects are terminated automatically. DFT cannot deal with the case that objects appear. However, it is free of pre-trained object detectors. 

![](https://i.imgur.com/T6epzVr.png)

Table 3 lists the major differences between DBT and DFT.

#### 2.2.2 Processing Mode

MOT can also be categorized into online tracking and offline tracking. 

The difference is whether or not observations from future frames are utilized when handling the current frame.
- Online, also called causal, tracking methods only rely on the past information available up to the current frame, 
- while offline, or batch tracking approaches employ observations both in the past and in the future.

##### Online tracking (=sequential tracking)

In online tracking [52], [53], [54], [56], [57], the image sequence is handled in a step-wise manner, thus online tracking is also named as sequential tracking.

An illustration is shown in Figure 2 (top), 
- with three objects (different circles) a, b, and c. 
- The green arrows represent observations in the past. 

The results are represented by the object’s location and its ID. Based on the up-to-time observations, trajectories are produced on the fly.


##### Offline tracking

Offline tracking [1], [46], [47], [51], [58], [59], [60], [61], [62], [63] utilizes a batch of frames to process the data. 

As shown in Figure 2 (bottom), observations from all the frames are required to be obtained in advance and are analyzed jointly to estimate the final output. 

Note that, due to computational and memory limitation, it is not always possible to handle all the frames at once. 

An alternative solution is to split the data into shorter video clips, and infer the results hierarchically or sequentially for each batch. 



![](https://i.imgur.com/W1oNuc5.png)

Table 4 lists the differences between the two processing modes


#### 2.2.3 Type of Output

This criterion classifies MOT methods into deterministic ones and probabilistic ones, depending on the **randomness of output**. 
- The output of deterministic tracking is constant when running the methods multiple times. 
- While output results are different in different running trials of probabilistic tracking methods. 


The difference between these two types of methods results from the optimization methods adopted as mentioned in Section 2.1.


#### 2.2.4 Discussion

The difference between DBT and DFT is whether **a detection model** is adopted (DBT) or not (DFT). 

The key to differentiate online and offline tracking is the way they process observations. 

Readers may question whether DFT is identical to online tracking because it seems DFT always processes observations sequentially. 
- This is true in most cases although some exceptions exist. 
- Orderless tracking [64] is an example.
- It is DFT and simultaneously processes observations in an orderless way. Though it is for single object tracking, it can also be applied for MOT, and thus DFT can also be applied in a batch mode. 


Another vagueness may rise between DBT and offline tracking, as in DBT tracklets or detection responses are usually associated in a batch way. 

Note that there are also sequential DBT which conducts association between previously obtained trajectories and new detection responses [8], [31], [65].

The categories presented above in Section 2.2.1, 2.2.2 and 2.2.3 are three possible ways to classify MOT methods, while there may be others. 

Notably, specific solutions for sport scenarios [5], [6], aerial scenes [44], [66], generic objects [8], [65], [67], [68], etc. exist and we suggest the readers refer to the respective publications.

By providing these three criteria described above, it is convenient for one to tag a specific method with the combination of the categorization label. 

This would help one to understand a specific approach easier.

---

## 3 MOT COMPONENTS


MOT의 목적은 연속된 프레임에서 각 물체를 추출 하고, 물체 식별자를 부여 하는 것이다. `In this section, we represent the primary components of an MOT approach. As mentioned above, the goal of MOT is to discover multiple objects in individual frames and recover the identity information across continuous frames, i.e., trajectory, from a given sequence. `

MOT 개발시 두가지 요소를 고려 해야 한다. `When developing MOT approaches, two major issues should be considered.`
- 프레임간 물체의 **유사도**를 어떻게 측정할 것인가? `One is how to measure similarity between objects in frames, `
- 프레임간 물체의 유사도를 기반으로 어떻게 **식별자**를 부여할 것인가? `the other one is how to recover the identity information based on the similarity measurement between objects across frames. `

Roughly speaking, 
- 유사도는 외형, 움직임등의 **모델링**에 관련된 일이다. `the first issue involves the modeling of appearance, motion, interaction, exclusion, and occlusion. `
- 식별자 부여는 **추론**에 관련된 일이다. `The second one involves with the inference problem. `


We review recent progress regarding both items as the following.


### 3.1 Appearance Model


### 3.2 Motion Model

#### 3.2.1 Linear Motion Model

#### 3.2.2 Non-linear Motion Model


### 3.3 Interaction Model (=mutual motion model)

Interaction model, also known as mutual motion model, captures the influence of an object on other objects. 

In the crowd scenery, an object would experience some “force” from other agents and objects. 
- For instance, when a pedestrian is walking on the street, he would adjust his speed, direction and destination, in order to avoid collisions with others. 

Another example is when a crowd of people walk across a street, each of them follows other people and guides others at the same time. 

In fact, these are examples of two typical interaction models known as 
- the social force models [100] and 
- the crowd motion pattern models [101].

#### 3.3.1 Social Force Models

#### 3.3.2 Crowd Motion Pattern Models.


### 3.4 Exclusion Model

Exclusion is a constraint employed to avoid physical collisions when seeking a solution to the MOT problem. 

이 가정은 물리적 환경에서 한 공간에 두 물체가 존재 할수 없기 때문이다. `It arises from the fact that two distinct objects cannot occupy the same physical space in the real world. `

Given multiple detection responses and multiple trajectory hypotheses,generally there are two constraints. 
- 탐지 단계에서의 배제 : 두개의 탐지값은 동일한 대상에 할당 되지 못한다. `The first one is the so called detection-level exclusion [105], i.e., two different detection responses in the same frame cannot be assigned to the same target. `
- 추적 단계에서의 배제 : 두 경로는 일치 할수 없다. `The second one is the so-called trajectory-level exclusion, i.e., two trajectories cannot be infinitely close to each other.`

#### 3.4.1 Detection-level Exclusion Modeling

Different approaches are adopted to model the detection level exclusion. Basically, there are “soft” and “hard” models.

##### “Soft” modeling.

##### “Hard” modeling


#### 3.4.2 Trajectory-level Exclusion Modelin


### 3.5 Occlusion Handling

![](https://i.imgur.com/t2reTSu.png)

**가려짐**은 MOT에서 가장 큰 문제점이다. 이로 인해서 ID 교환이나 경로 단절 등의 문제가 발생 한다. 다양한 해결 방법이 제안 되었다. `Occlusion is perhaps the most critical challenge in MOT.It is a primary cause for ID switches or fragmentation of trajectories. In order to handle occlusion, various kinds of strategies have been proposed`


#### 3.5.1 Part-to-whole

가려짐이 발생 하여도 일부분은 보인다는 가정하에 진행 된다. 보이는 일부분으로 전체 모습을 추론 하여 동작 한다. `This strategy is built on the assumption that a part of the object is still visible when an occlusion happens. This assumption holds in most cases. Based on this assumption, approaches adopting this strategy observe and utilize the visible part to infer the state of the whole object. `

The popular way is dividing a holistic object (like a bounding box) into several parts and computing affinity based on individual parts. If an occlusion happens, affinities regarding occluded parts should be low. Tracker would be aware of this and adopt only the un-occluded parts for estimation. Specifically, parts are derived by dividing objects into grids uniformly [52], or fitting multiple parts into a specific kind of object like human, e.g. 15 non-overlap parts as in [49], and parts detected from the DPM detector [110] in [77], [111].

Based on these individual parts, observations of the occluded parts are ignored. For instance, part-wise appearance model is constructed in [52]. Reconstructed error is used to determine which part is occluded or not. The appearance model of the holistic object is selectively updated by only updating the unoccluded parts. This is the “hard” way of ignoring the occluded part, while there is a “soft” way in [49]. 

Specifically, the affinity concerning two tracklets j and k is computed as P where f is feature, i is the index of parts. 

The weights are learned according to the occlusion relationship of parts. In [77], human body part association is conducted to recover the part trajectory and further assists whole object trajectory recovery. “Part-to-whole” strategy is also applied in tracking based on feature point clustering, which assumes feature points with similar motion should belong to the same object. As long as some parts of an object are visible, the clustering of feature point trajectories will work [62], [68], [112].

#### 3.5.2 Hypothesize-and-test


This strategy sidesteps challenges from occlusion by hypothesizing proposals and testing the proposals according to observations at hand. As the name indicates, this strategy is composed of two steps, hypothesize and test. 

##### Hypothesize. 

Zhang et al. [38] generate occlusion hypotheses based on the occludable pair of observations, which are close and with similar scale. 


##### Test

The hypotheses would be employed for MOT when they are ready. Let us revisit the two approaches described above. 

In [38], the hypothesized observations along with the original ones are given as input to the cost-flow framework and MAP is conducted to obtain the optimal solution. 

In [114] and [113], a multi-person detector is trained based on the detection hypotheses. 

This detector greatly reduces the difficulty of detection in case of occlusion.

#### 3.5.3 Buffer-and-recover

This strategy buffers observations when occlusion happens and remembers states of objects before occlusion. 

When occlusion ends, object states are recovered based on the buffered observations and the stored states before occlusion.

Mitzel et al. [71] keep a trajectory alive for up to 15 frames when occlusion happens, and extrapolates the position to grow the dormant trajectory through occlusion. 

In case the object reappears, the track is triggered again and the identity is maintained. This idea is followed in [34]. Observation mode is activated when the tracking state becomes ambiguous due to occlusion [115]. 

As soon a enough observations are obtained, hypotheses are generated to explain the observations. This could also be treated as “buffer-and-recover” strategy

#### 3.5.4 Others

The strategies described above may not cover all the tactics explored in the community. 

For example, Andriyenko et al. [116] represent targets as Gaussian distributions in image space and explicitly model pair-wise occlusion ratios between all targets as part of a differentiable energy
function. 

In general, it is non-trivial to distinctly separate or categorize various approaches to occlusion modeling, and in some cases, multiple strategies are used in combination.

---

## 3.6 Inference

> 프레임간 물체의 유사도를 기반으로 어떻게 **식별자**를 부여할 것인가?

### 3.6.1 Probabilistic Inference


#### Kalman filter


#### Extended Kalman filter


#### Particle filter


### 3.6.2 Deterministic Optimization

#### Bipartite graph matching

#### Dynamic Programming

#### Min-cost max-flow network flow

#### Conditional random field

#### MWIS



### 3.6.3 Discussion

현실에서는 확률적 방법보다 결정론적 방법이 더 많이 사용된다. `In practice, deterministic optimization or energy minimization is employed more popularly compared with probabilistic approaches. `

비록 확률적 방법이 좀더 직관적이고 완벽한 해결책으로 보이지만 대부분의 경우 infer가 제대로 수행되지 않는다. `Although the probabilistic approaches provide a more intuitive and complete solution to the problem, they are usually difficult to infer. `

반면에 **energy minimization**는 제한된 시간동안 나쁘지 않은 결과를 도출 해낸다. `On the contrary, energy minimization could obtain a “good enough” solution in a reasonable time.`













