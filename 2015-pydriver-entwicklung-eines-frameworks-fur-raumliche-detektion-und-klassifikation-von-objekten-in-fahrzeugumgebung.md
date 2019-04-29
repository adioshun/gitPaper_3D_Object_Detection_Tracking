http://lpltk.github.io/pydriver/index.html

PyDriver: Entwicklung eines Frameworks für räumliche Detektion und Klassifikation von Objekten in Fahrzeugumgebung. 



#

## 목차 


1 Introduction
1.1 Motivation. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 5
1.2 Goal of the work. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 6
1.3 Approach. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 7
1.3.1 Tools. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 7
1.3.2 Structure. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 8th

2 Theoretical basics
2.1 Object detection. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 10
2.2 Preprocessing. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 11
2.2.1 Stereo recordings. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 11
2.2.2 Ground level. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 12
2.3 Keypoints. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 13
2.3.1 Harris detector. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 13
2.3.2 Intrinsic Shape Signatures. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 14
2.4 features. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 15
2.5 Classification and Object Parameter Estimation. , , , , , , , , , , , , , , , , , , , , , , , 16
2.6 Evaluation measures. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 17


3 realization
3.1 Structure. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 21
3.2 Tools. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 23
3.2.1 Cython and NumPy. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 23
3.2.2 scikit-learn. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 24
3.2.3 OpenCV and ELAS. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 25
3.2.4 OpenCL. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 25
3.2.5 Point Cloud Library. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 26
3.3 Data Structures. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 28
3.4 Pipeline. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 30
3.4.1 Records. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 30
3.4.2 Preprocessor. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 31
3.4.3 Keypoints and Features. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 34
3.4.4 Detection. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 35
3.4.5 Evaluation. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 37
3.4.6 Visualization. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 38
3.5 Optimization. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 41
3.5.1 Multithreading. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 41
3.5.2 GPU. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 42
3.5.3 Ordered Point Clouds. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 43
3.6 Documentation. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 44

4 evaluation and results
4.1 Validation of the general functionality of the framework. , , , , , , , , , , , , , , 45
4.2 Features of the framework. , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 49
5 Summary and outlook 53
References 55
List of Figures 61
---


1.3 Approach
1.3.1 Tools
In order to enable rapid prototype development, higher-level programming languages ​​are suitable
like Java, Python, Ruby, or even specialized environments like MATLAB. The
chosen language should be freely available and widely disseminated to facilitate familiarization,
offer fast numerical calculations and be portable.
Python meets these criteria. This language has mature libraries
for scientific computing and is already being used extensively in research [Mil11],
It was chosen as the main programming language of the framework. Another advantage is the
Possibility to include Python modules in C ++ programs. However, a disadvantage of Python is
the comparatively slow execution, if the use of existing optimized
Packages is not possible. This is done in the framework through the use of Python language extension
Cython [Beh11] balanced. With the integration of OpenCL, the use of
GPUs made easier [DP14] and demonstrated in the framework.
A major hurdle for the use of Python for the processing of point clouds is the
The fact that the extensive library »Point Cloud Library« (PCL) is not available
Python interfaces and is designed exclusively for C ++. The very basic for PCL
Use of C ++ templates makes their use in a dynamically typed language difficult
Python in addition.
As a solution, a specialized Python interface for PCL is developed, which is based on the
mework required functions of the PCL library is limited and expandable as needed. These
represents an additional layer through which also the need for a PCL installation
Target computers with a Windows operating system are avoided by using static binding,
which greatly simplifies the installation process of the framework.

1.3.2 Structure
The framework should facilitate the following workflow:
1. Reconstruction of a point cloud from lidar data, stereo images or other sources
2. Detection and removal of the ground plane
3. Determination of keypoints in the point cloud
4. Extraction of features at the keypoints
5. Training of machine learning
6. Evaluation
Individual steps are optional and can be omitted if necessary, as well as the distance
the ground level, which may not be necessary depending on the procedure.
The user has the opportunity to use the object-oriented design of the framework
extend or add new components such as keypoint detectors and feature extractors
to implement. As long as the interfaces of the modified components to the framework
remain patible, they can be seamlessly integrated. The framework remains flexible
and leaves the user free space for deviations from the above-mentioned workflow or
the existing interfaces, while the necessary adjustments through dynamic
Data structures are reduced.
The framework also provides an interface for linking to benchmark data sets
offered and the user through visualization functions in his work supported. The connection
to the data sets »Object« and »Tracking« of the »KITTI Vision Benchmark created by KIT
Suite «[Gei13] is implemented.



## 2.2 Preprocessing
The necessary or desirable preprocessing of the data depends greatly on the type and
Quality of the sensors used. The whole range of possible preprocessing steps
can not cover the framework, but some essential features are offered.
Two of them, the deep reconstruction of stereo recordings and the processing of the ground level,
will be explained in more detail here.
2.2.1 Stereo recordings
For approaches based on optical stereo recordings, the depth information is not
directly in front. To obtain these, both camera perspectives must be combined and shared12
2 Theoretical basics
be evaluated. 1 By the distance between the two cameras (called "baseline")
creates an angle between the lines that the respective camera with the object or a
connect the corresponding point. The angle between these lines is called parallax
and provides an apparent shift of the imaged point when comparing the two
Recordings. The parallax depends on the distance of the imaged point, it is larger
for near and zero for infinitely distant points. If the baseline is known, can from the
Parallax the distance of the point can be determined.
To reconstruct the entire visible scene three-dimensionally becomes well identifiable
Wanted areas in both recordings. The distance of their pixel coordinates, called disparity,
is directly influenced by the parallax and flows into the disparity map, which is made up of
the evaluation of the entire image results. If the cameras are calibrated, out of the disparity
the distance will be calculated. Since the calculation of the disparity is subject to errors and by
Obscurations, lack of uniqueness or even runtime restrictions not for all pixels
is possible, then further processing steps are necessary.
To implement this approach, there are various methods, including, but not limited to
distinguish how appropriate image areas are searched for and identified and whether the result
optimized locally or globally. Two such methods are provided in the framework.
One of them is implemented in the OpenCV library and is based on some modifications
the semiglobal matching method [Hir08], whereby u. a. the similarity measure from [Bir98] is used.
As an alternative, the ELAS library [Gei10] is integrated.

2.2.2 Ground level
Determining the ground level is essential for the perception of the environment in traffic,
this limits the search spaces for relevant objects and changes in position (nod and
Roll) of the own vehicle can be compensated. That the floor is level is there
a simplifying assumption that is valid at least in the immediate vicinity of the vehicle.
For estimating the parameters of a plane present in the data in the presence of
Outliers and other objects can be analyzed using the Random Sample Consensus method (RANSAC),
originally introduced in [Fis81]. At first a subset of the

entire point cloud selected at random and one level adapted to it. All points of the original
a cloud whose distance from the plane does not exceed a threshold will be included
referred to as inliers and form the consensus set. These steps are repeated iteratively and the
Plane with the largest consensus set is considered the best estimate. The parameters of the level
can be refined by re-fitting to the determined consensus set.
The framework uses the implementation of this method in the Point Cloud Library [Rus11],
additional constraints can be made on the location of the layer. So
The process can be accelerated and the result improved by limits for inclination
and vertical position of the plane, and thus poorly fitting parameter hypotheses
be excluded early. In particular, this is the adaptation of the level to others
Avoid surfaces such as building facades or tunnel ceilings.
After estimating the ground level parameters, the level can be removed from the framework and
the point cloud can be brought into a predetermined position by an affine image, so that the
detected ground plane on the horizontal plane in the coordinate system comes to rest. The
Point of the ground plane, which is vertically below (or above) the origin with respect to the ground plane
of the sensor coordinate system, becomes the new coordinate origin. Overall, you can
thus shifts and rotations of the point cloud in vertical vehicle movements as well
Roll and pitch turns are minimized.


3.4.2 Preprocessor
The preprocessing is done by a class based on the processing
interface and designed to meet the needs of the user
is configured. The configuration takes place by the transfer of several objects, which for the
Construction of the point cloud and its preprocessing are responsible.

When processing a recording, the processing routine of the reconstructor
Object executed. There are two implementations for this, the lidar data or stereo
use ras. Then the list of data processor objects in the given
Order processed. Both for the point cloud reconstruction and its subsequent
de preprocessing are defaults defined on the processing of stereo recordings
are tuned to facilitate familiarization and initial experimentation with the framework.
All objects receive complete information about the scene (and not exclusively
the point cloud), which are also stored in a dictionary, and can be any
manipulate, so in addition to the change of existing values ​​and objects also completely new
Add information.
This object-oriented configuration provides sufficient flexibility to own
Reconstructors and data processors to be able to implement without doing a special
Having to write a preprocessor as long as the desired procedure
programmed schema. Another advantage is that the configuration is thus hierarchical
and the configuration possibilities of the preprocessor are not limited to new subobjects.
must be adapted, since these are already transferred by the
Application were configured.
The preprocessor also implements the possibility of pre-processing results
permanently store, eliminating repetitive runs and evaluations of records
be greatly accelerated as long as the preprocessing steps remain constant. This is going through
The option described in section 3.3 achieves the objects created in the framework
serialize.

data processors

The Framework contains some implementations of data processors, which in particular include the
Functions of downsampling, viewport constraint, ground plane processing,
implement and remove invalid values. All data processors are from one
derived common base class, which the developer in the implementation of its own processors
supported.
During downsampling, the resolution of the point cloud is scaled down. For this, the PCL
Functionality that covers all of the voxel-covered space in Voxel's
divide and approximate the points contained therein (if any) by the single point, whose coordinates are determined from the geometric center of gravity of the original points
the. This reduces the number of points, especially in denser areas of the point cloud,
while voxels are unaltered with just one point, so further processing will help
moderate information loss can occur faster.
The display field restriction is used to limit the point cloud to specified inter-
valle within the three spatial axes. This can increase the performance of other functions
become less interesting areas that are further away or one particular
Height above ground, be removed.
The processing of the ground level includes their detection and optionally their removal and
the transformation of the point cloud into a position specified relative to the ground plane (see section
2.2.2). Various parameters can be set here, one of which is the limitation of the situation
the detected level and the accuracy and speed of detection
control by limiting the point cloud to a certain area before detection and
then scaled down.
The removal of invalid values ​​leads an ordered point cloud (see section 3.5.3) into an unused point cloud.
rearranged point cloud. If further processing needs no order, then the application is
This data processor makes sense, since the processing of the actual points on this
Way is accelerated.

Reconstruction from stereo recordings

The stereo reconstruction is controlled by a Reconstructor class that has one or more
Takes matcher classes as configuration. A single matcher calculates the disparity card
based on two images. He has further attitudes, which depend on the concrete
used to depend on algorithm and, for example, can determine the size of windows
which he operates on. The three matchers included in the framework provide interfaces to OpenCV and
the ELAS library (see Section 3.2.3), where two of these matchers use ELAS. One of them
converts the color space of the images to grayscale while the other reconstructs separately
allowed on the basis of individual color channels.
However, the results of a single matcher can also be ignored
Problems such as occlusions be patchy, z. B. then, if the finer reconstruction within
smaller window in roughly textured areas is not possible. To address this problem,
The Reconstructor class provides an interface to merge the results of multiple matchers.
In the framework, a simple procedure is implemented by putting all matchers in the given
Sequence and invalid values of the disparity card iteratively by their results
replaced as soon as one of the matchers produces valid values in corresponding places.

Reconstruction from Lidardaten

The reconstruction from lidar data essentially involves reading in the delivered spatial data.
point coordinates and is thus simpler. For coloring the dots,
if available, use the reflectance information of the lidar sensor, which only
te supplies. In addition, optional projection into the available camera shots can be done.
In this case, the points that are visible in at least one of the shots, with the
Color information of the corresponding pixel colored.
Since the KITTI dataset contains only annotations for the objects visible on the images,
It is also possible to remove all lid points not visible on the images. To change-
if objects outside the camera could be recognized correctly, but in the evaluation as
Phantom objects (false positives).

