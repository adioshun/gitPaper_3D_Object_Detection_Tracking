# Car2X-based perception in a high-level fusion architecture for cooperative perception systems

https://ieeexplore.ieee.org/document/6232130



In cooperative perception systems, different vehicles share object data obtained by their local environment perception sensors, like radar or lidar, via wireless communication. 

In this paper, this so-called Car2X-based perception is modeled as a virtual sensor in order to integrate it into a highlevel sensor data fusion architecture. 

The spatial and temporal alignment of incoming data is a major issue in cooperative perception systems. 

시간 정렬:Temporal alignment is done by predicting the received object data with a model-based approach. 
- In this context, the CTRA (constant turn rate and acceleration) motion model is used for a three-dimensional prediction of the communication partner’s motion. 

공간 정렬: Concerning the spatial alignment, two approaches to transform the received data, including the uncertainties, into the receiving vehicle’s local coordinate frame are compared. 
- The approach using an unscented transformation is shown to be superior to the approach by linearizing the transformation function. 
- Experimental results prove the accuracy and consistency of the virtual sensor’s output.