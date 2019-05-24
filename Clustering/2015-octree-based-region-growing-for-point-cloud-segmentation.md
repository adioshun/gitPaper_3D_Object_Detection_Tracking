# [Octree-based region growing for point cloud segmentation](https://www.researchgate.net/publication/274645446_Octree-based_region_growing_for_point_cloud_segmentation)

1. Model fitting-based methods
2. Region growing-based methods
3. Clustering feature-based methods


## 1. Model fitting-based methods(=parameter-based approach)

### 1.1 Hough Transform (HT) (1981)

The HT is used to detect 
- planes (Vosselman et al., 2004), 
- cylinders (Tarsha-Kurdi,2007), 
- spheres (Rabbani and Heuvel, 2005). 

속도/신뢰성 향상을 위해 파라미터 선별시 여러 Step 으로 나누어 진행 `determined the parameters of the objects through several separate steps. `

For example, plane identification employed two steps: 
- (1) determination of the plane normal vector and 
- (2) establishment of the distance from the plane to the origin.

### 1.2 the Random sample consensus (RANSAC) (1981)

동작 원리 : The RANSAC paradigm is used to extract shapes by 
- randomly drawing minimal data points to construct candidate shape primitives. 
- The candidate shapes are checked against all points in the data set to determine a value forthe number of the points that represents the best fit.

#### 성능 향상 방안 #1

속도 향상을 위해 octree사용 `An octree was employed for efficiently extracting sampling points.` 
- 수 밀리온 포링트를 1분 안에 처리, 수시간 -> 수초로 단축 

#### 성능 향상 방안 #2

Similarly, Chen et al. (2014) improved the RANSAC algorithm through the localized sampling to segment the polyhedral rooftop primitives and then through the application of a region growing based triangulated irregular net-work (TIN) to separate the coplanar primitives. 

#### 성능 향상 방안 #3

In related work, Awwad et al. (2010) modified the RANSAC algorithmby using a clustering technique to divide the dataset into small clusters basedon normal vectors of the points. 

#### 성능 향상 방안 #4

To prevent spurious surfaces appearing in a group, a sum of the perpendicular distance between the points and a lo-cal surface was imposed as a condition to decide on whether the plane be accepted within the group or eliminated as being outside of the group.


### 1.3 단점 

1. First, since these algorithms only use point positions, many spurious planes that do not exist in reality may be generated. 

2. Second, the segmentation quality is sensitive to the point cloud characteristics (density, positional accuracy,
and noise). 

3. Third, the algorithms perform poorly with large datasets or those with complex geometries. 
    - HT especially requires significant processing time and high memory consumption for large data sets, because all parameters must be stored. 
    - Furthermore, HT is very sensitive to the selection of surface parameters (Awwad et al., 2010; Tarsha-Kurdi, 2007).
    

## 2. Region growing-based methods (1998)
    
The method involved two stages
- a coarse segmentation based on the mean and Gaussian curvature of each point and its sign, and a refinement using an iterative region growing based on a variable order bivariate surface fitting. 
- then adopted by others for 3D point cloud segmentation. 
    - eg, Gorte (2002) performed a region growing segmentation using a TIN as the seed surface and the angle
and distance between the neighboring triangles for the growing. The seed region was used to merge the triangles. 
    - eg, In contrast, Tóvári and Pfeifer (2005) used normal vectors, the distance of the neighboring points to the adjusting plane, and the distance between the current point and candidate points as the criteria for merging a point to a seed region that was randomly picked from the data set after manually filtering areas near edges. 
    - eg. Nurunnabi et al. (2012) also used these criteria but with the seed points being those having the least curvature.
    
    
As the points in the interior region have been shown to be good seed points, Vieira and Shimada (2005) firstly removed points along sharp edges using a curvature threshold. Median filtering was then performed to reduce noise and the remaining points were considered as seed points. 

Rabbani et al. (2006) proposed as an alternative the residual of a plane fitting to select the seed points followed by region growing using an estimated point normal and the residual. 

Ning et al. (2009) proposed a two-step region growing segmentation: 
- rough segmentation to extract points on the same plane based on normal vectors followed by fine segmentation to extract architectural details based on the residual that is the distance from the point to the local shape. 

효율성/강건성을 위해서 `To improve efficiency and robustness of the region growing method, `
- Deschaud and Goulette (2010) introduced the area of the local plane as a criterion for selecting the seed region and then employed an octree to search the neighboring points of those that should possibly be merged to the seed region.


Region growing based on octrees or voxel grids have been introduced to improve efficiency. 

- eg. Woo et al. (2002) to recursively divide a dataset into smaller voxels until the standard deviation of voxels’ normal vectors was less than the threshold, where the voxel’s normal vector is the average the point normal vectors. 
    - In this approach, the segmentation extracted edge-neighborhood points possessed by the voxels having the normal vector when the size was smaller than a predefined cell size threshold. 
    - The adjacent voxels were then merged to the leaf node, if the deviation of the voxels’ normal vectors was less than the tolerance. 

- eg. Similarly, Wang and Tseng (2004) used the residual distance and an area of the data points within the voxels as
the criteria for subdividing an initial bounding box. 
    - The voxels on the same layer having similar normal vectors were classified as being in the same group,where a normal vector of the voxel was computed from data points within the voxel. 
    
- eg. Subsequently, Wang and Tseng (2011) proposed splitting and merging algorithms to extract coplanar points from a connected voxel group.
    - The region growing was then used to merge a group of coplanar points based on the angle variation of the fitting planes amongst them. 
    - In these works, the voxel size and how the normal vector of the voxel was computed strongly influenced the segmentation results.


### 단점 

1. they are not particularly robust as has been shown experimentally in part 
    - because the segmentation quality strongly depends both on multiple criteria and the selection of seed points/regions, where no universally valid criterion exists. 

2. Additionally, they also require extensive computing time when 3D point clouds are used 


## 3. Clustering feature-based methods


Another major segmentation approach employs clustering of features.

- Filin (2002) employed the formation of a feature space and a mode-seeking algorithm based on seven point-based parameters to extract point cloud surface classes. 
    - The clustering in the feature space excluded the boundary points and, thus, a refinement phase was necessary to test whether the point was within the same cluster. 

- Biosca and Lerma (2008) used a fuzzy clustering segmentation method with six point-based parameters and refined the clusters by merging unlabeled points into the nearest cluster, if the distance between the points to the plane was less
than a pre-specified threshold. 

- Hofmann (2003) introduced clustering methods based on feature vectors computed from a TIN-structure for both 2D and
3D problems. 
    - The feature vectors were triangle-mesh slope and orientation for 2D problems. 
    - Where for 3D, O-distance was used in addition to the two features. 
    - The O-distance was defined as the minimum distance of a plane calculated from the origin. 
    - The authors concluded a prior knowledge of the data accuracy is important for a successful clustering and that single outliers do not affect the cluster analysis.

As the choice of neighborhood strongly influences estimated feature vectors, 

- Filin and Pfeifer (2006) used a slope adaptive neighborhood to compute the point features and a mode-seek algorithm to detect the points in the surface class. 
    - The extracted clusters were then refined by merging them with adjacent clusters that shared common attributes based on their neighborhood relationships. 
    
- Lerma and Biosca (2005) also used a mode-seek algorithm to separate data points belonging to planar surfaces after using a fuzzy C-means (FCM) algorithm to classify the point into various categories based on six point-based parameters. 
    - The real regions are obtained by employing a region-growing algorithm to separate non-parallel planes. 
    
- Finally, in another way, Dorninger and Nothegger (2004) used the hierarchical clustering technique in four dimensional feature spaces to extract the seed clusters for region growing. 
    - The points in a region around the seed cluster are merged to the seed cluster, if the normal distance from the points to the seed cluster is smaller than a predefined threshold.

### 장/단점 

1. 강건하지만 계산 부하가 크다. `The clustering techniques for segmenting are robust methods that do not require seed points/regions, unlike the region growing-based methods, but clustering methods are computationally expensive for multi-dimensional features in large datasets.`


2. 파라미터(Knn크기, 노이즈 레벨)의 영향이 크다. `Also, the method’s results depend on the quality of point feature computation, which is strongly affected by the selection of the neighborhood size and the data’s noise level. `


3. Furthermore, the clustering methods developed to date cannot detect consistently data points around edges, 
    - because feature vectors of these points often differ from other points of the surface classes. 
    - Thus, their final segmentation results depend on a refinement phase, where the parallel clusters must be separated and the edge points must be merged onto the clusters.

