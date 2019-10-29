# [Multiple Object Tracking: A Literature Review](https://arxiv.org/pdf/1409.7618.pdf)



MOT는 오랜 기간 연구되었지만 아직 부족하다. 본 논문을 통해 리뷰를 하려 한다. `Multiple Object Tracking (MOT) is an important computer vision problem which has gained increasing attention due to its academic and commercial potential. Although different kinds of approaches have been proposed to tackle this problem, it still remains challenging due to factors like abrupt appearance changes and severe object occlusions. In this work, we contribute the first comprehensive and most recent review on this problem. We inspect the recent advances in various aspects and propose some interesting directions for future research. To the best of our knowledge, there has not been any extensive review on this topic in the community. We endeavor to provide a thorough review on the development of this problem in recent decades.` 

The main contributions of this review are fourfold: 
- 1) Key aspects in a multiple object tracking system, including formulation, categorization, key principles, evaluation of an MOT are discussed. 
- 2) Instead of enumerating individual works, we discuss existing approaches according to various aspects, in each of which methods are divided into different groups and each group is discussed in detail for the principles, advances
and drawbacks. 
- 3) We examine experiments of existing publications and summarize results on popular datasets to provide quantitative comparisons. We also point to some interesting discoveries by analyzing these results. 
- 4) We provide a discussion about issues of MOT research, as well as some interesting directions which could possibly become potential research effort in the future.


## 1 INTRODUCTION

Multiple Object Tracking (MOT), or Multiple Target Tracking (MTT), plays an important role in computer vision. The task of MOT is largely partitioned to locating multiple objects, maintaining their identities, and yielding their individual trajectories given an input video. Objects to track can be, for example, pedestrians on the street [1], [2],vehicles in the road [3], [4], sport players on the court [5], [6], [7], or groups of animals (birds [8], bats [9], ants [10],fish [11], [12], [13], cells [14], [15], etc.). 
Multiple “objects” could also be viewed as different parts of a single object[16]. 

In this review, we mainly focus on the research on pedestrian tracking. The underlying reasons for this specification are threefold. First, compared to other common
objects in our environment, pedestrians are typical nonrigid objects, which is an ideal example to study the MOT
problem. Second, videos of pedestrians arise in a huge number of practical applications, which further results in great
commercial potential. Third, according to all data collected
for this review, at least 70% of current MOT research efforts
are devoted to pedestrians