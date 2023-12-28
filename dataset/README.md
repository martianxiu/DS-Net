# About the dataset

The original LiDAR scan is available on [OpenTopography](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.052018.2444.1). 

The dataset used in this study contains 6000+ samples ("building patches") and each sample is assigned a unique ID. Each sample consists of several attributes:
* 'building_mask': a binary mask that assigns 1 to the points of the central building and 0 to the other.
* 'damage': an integer that indicates the damage level of the central building. 0 for Non-collapsed and 1 for Collapsed. 
* 'intensity': backscattered intensity of each point. 
* 'per_point_labels': a mask that assigns the damage level to each point in a sample.
  * 0: Background
  * 1: Non-collapsed
  * 2: Collapsed  
* 'points': 3D coordinates of each point.

In this study, 'points' is used for the input and 'per_point_labels' is used for the target.
