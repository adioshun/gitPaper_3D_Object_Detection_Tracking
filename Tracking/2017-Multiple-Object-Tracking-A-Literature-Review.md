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






























