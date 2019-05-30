# [Effective Data Association Algorithms for Multitarget Tracking](https://macsphere.mcmaster.ca/bitstream/11375/16272/2/thesis%20-%20Biruk%20Habtemariam.pdf)

## 2. Background

### 2.1 System Model


#### 2.1.1 Target Dynamics


비행기 같은 physical systems의 dynamics는 **state-space model**로 모델링 가능 하다. `The dynamics of physical systems such as airplanes, vessels, ballistic targets, etc., can be modeled using a state-space model [14][18][93]. `

With state-space modeling, a moving target can be represented as the transition in the state of a target driven by a process noise.

##### A. nonlinear model



##### B. linear time invariant system


#### 2.1.2 Observation Model


### 2.2 TBD vs DBT

센서 값에서 타겟의 상태를 추정 하는 방법은 2가지가 있다. `estimate the target state from the observation data`
- TBD
- DBT 


#### A. Track-Before-Detect (TBD)

-Raw신호를 직접 처리 하여 Track의 패턴을 찾는 방법 `One method to estimate the target state from the observation data is to directly process the raw signal and search for track patterns. `

- 신호가 약하고 배경 노이즈와 구분이 어려운 SNR시나리오에 적합 `This approach is effective for low Signal-to-NoiseRatio (SNR) scenario where signal from a target is weak and indistinguishable from the background noise [94]. `

- TBD methods use the entire measurement set of a sensor’s resolution cells and integrate tentative targets over multiple frames [95]. 

- 계산 부하가 크다. `As a result, TBD methods are computationally demanding in most cases.`


#### B. Detect-Before-Track

- 탐지 정보를 활용 `Another approach is to apply a threshold to the raw signal from the radar and extract the measurements (also referred as detection, contacts and radar plots in the literature) [14]. `

- 추출된 Measurement에서 track과 데이터 연동 작업을 수행 `Based on the extracted measurements, new tracks can be initialized or already initialized tracks can be associated to measurements and their states be updated using the associated measurements [18]. `

- As detection precedes the tracking process, this methods are collectively referred as Detect-Before-Track.

### 2.3 Data Association

DBT의 첫 단계는 radar에서 들어오는 raw 신호에 임계치를 적용 하는 것이다. `As discussed in the previous section, the initial step in Detect-Before-Track methods is to apply a threshold to the raw signal received from the radar. `

일반적인 임계치 방법은 **CFAR**이다. `The common thresholding technique is the Constant False Alarm Rate (CFAR) method [31]. `

이 방식은 적응형 방식으로 데이터에 따라 값이 다르다. `The CFAR is an adaptive thresholding approach, in which a constant false alarm rate can be achieved by applying a threshold level determined by sliding window neighbourhood resolution cells averaging technique [72]. As a result, the threshold level can go up and down from one resolution cell to another depending on the local clutter situation.`

Applying thresholds results in a discrete measurement set either from target or clutter distributed in the entire measurement space. 

따라서 측정값이 tart에서 온것인지, clutter에서 온것인지 측적값과 track을 연결하는 작업이 필요 하다. `Hence, a measurement-to-track data association is required to determine if a measurement is from a target or clutter [14]. `


대부분의 DA기술들은 계산 부하를 중이기 위해 Most data association techniques involve gating techniques in order to reduce the number of feasible measurement-to-track association. 

If a track’s previous state and covariance is known, a measurement validation gate can be constructed around the predicted track position. 

The simplest method is to specify a regular region that will satisfy the gate requirement [18]. 

A more effective approach is to define an n-dimensional ellipse around the predicted track position and choose measurements that satisfy the condition



