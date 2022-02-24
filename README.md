# Fine-grained Localisation Models by Image Matching

## About the project
In this work, we introduce several approaches to identify fine-grained locations from 2D images. The methods firstly shortlist the candidate images based on the similarity of the extracted features from pre-trained Vision Transformer (ViT) [1] to speed up the process. After then, they select the best match image sharing the most number of confident matching points from pre-trained LoFTR [2]. The geographic information of the best image represents the input image location. Different approaches involve different techniques such as the image transformation or the result adjustment for better accuracy. We evaluate the performance of the different methods through the experiments and show they can plausibly recognise the positions with unseen images.

## Dataset (not provided in this repo)
The total number of 7,500 training images and 1,200 test images are used. Each of which are taken at an art gallery in 680 (W) x 490 (H) pixels. Training images have geographic information, x and y, values from a mapping algorithm. For simplicity, we assume there is no radial distortion and ignore other artefacts and distortion on the images. In order to select the best model, we compare the performance on initially split validation dataset that contains 750 images from the original training images. 

## Evaluation Metric
Mean Absolute Error (MAE) between the predicted and the true geographic coordinates.

## Approaches
**A. Top 1 ViT-similar** directly uses the coordinates of the most similar image in train set (top 1 ViT-similar) as the prediction. 

**B. Top 10 ViT-similar & most LoFTR matching point** 
1) Shortlist top 10 similar images in the train set
2) Count the number of matching points from LoFTR between the test image and each of the 10 images, select one image with the most counts
3) Predict the location as the same x,y coordinates of the best matching train image

**C. Affine Transformation & Linear Regression**
1) Shortlist top 10 similar images in the train set
2) Count the number of good points (over 0.7 confidence) from LoFTR, select one image with the most counts
3) Randomly select three points from the confident points and generate affine matrix. There is no output when it fails to find the matrix elements
4) Train a linear regression (w/ default setting) with affine elements (as features) and difference between the true location and the selected image location

**D. Camera Transformation - Essential Matrix**
1) Shortlist top 10 similar images in the train set
2) Find the essential matrices between every 10 pairs with LoFTR detected matching points 
3) Select one image with the most inlier points from its corresponding essential matrix

**E. Camera Transformation - Camera Pose**
1) Shortlist top 10 similar images in the train set
2) Find the essential matrices between every 10 pairs with LoFTR detected matching points 
3) Decompose the essential matrices into the rotation and translation matrices
4) Select one image with the most points from LoFTR that can be described by the two transformations (rotation and translation)

## Evaluation on Test dataset
| Approaches | A | B | C | D | E |
| ---------- | - | - | - | - | - |
| MAE (test)| 7.01 | 6.14 | 5.86 | 4.35 | 4.37 |

Those approaches show a competitive performance on Kaggle Competition

## Version
Torch 1.7.1<br>
Torchvision 0.10.1<br>
PIL 8.3.1<br>
Pandas 1.3.1<br>
Numpy 1.19.5<br>
Matplotlib 3.4.2<br>
OpenCV 4.5.2<br>

## References
[1] A. Dosovitskiy et al., ”An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020. <br>
[2] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, ”LoFTR: Detector free local feature matching with transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 8922-8931


-------


