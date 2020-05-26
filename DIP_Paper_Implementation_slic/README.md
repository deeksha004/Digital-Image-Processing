This repository provides a pipeline to implement paper named as SLIC Superpixel Methods Compared to state of the art superpixel methods.

Python3 is supported.

## Install

Make sure you have a working versions of skimage, PIL, openCV2, pandas, numpy, matplotlib

### Ubuntu 16.04

To Implement Existing Superpixel Algorithms

1. Clone the repository
2. Implement Existing Superpixel Algorithms
`python3 slic_imp.py input-images-path number-of-segements compactness`
3. Image segmentation using DIP without ML
`cd image_segmentation_without_ml`
`python3 slic_seg.py images/car.jpg masked/car.jpg number-of-segements compactness`
4. Image segmentation using DIP and ML
Run the below commands. (Second Argument is the trained model name i.e 1. seg_rf_model_original, 2. seg_rf_model_slic 3. seg_rf_model_Felzenswalb)
`cd image_segmentation_with_ml`
`python3 predict.py images/test/*.jpg seg_rf_model_original`

