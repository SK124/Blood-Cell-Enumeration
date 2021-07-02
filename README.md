TLDR;

Object Enumeration of minute and closely packed entitites using YOLOv3 & v4 using K means clustering for anchor box calculation on BCCD Datatset.Perfomance and the data modelling process and prospects of the planned work for next month has been documented. 


**BLOOD CELL DETECTION AND ENUMERATION**

**Abstract :**

Blood Cell consists of RBCs, WBCs, Platelets and their respective subtypes, everytime doctors need a report, lab technicians take hours to count and estimate and the report takes a day or two to arrive in some cases. Recent developments in Object detection algorithms have paved the way to automate many computer vision tasks detection in particular such as vehicle detection, which would not have been possible with traditional convolutional networks. This concept could be used in biological discipline where automation is scarce and in dire need. We are planning to build a novel object detection architecture specific for identifying and accurately enumerating minute and densely packed entities on microscopic level.

**Approach :**

We started with the standard object detection algorithms such as Faster RCNN, YOLOv3, SSD. We understood their architecture and implemented them on autonomous driving datasets like KITTI, BDD100k and understood their pros and cons such as ease of preprocessing, accuracy, speed and complexity of the network. Our implementation on these datasets can be found [here](https://colab.research.google.com/drive/1iOjUY32E9Or2g5wWXwkNrC-sPpv14fu3). We understood that our use case demands more accuracy and could trade off speed for better precision.

In the next step we decided to implement these models on the Blood Cell Dataset and train them from scratch and compare their performances. Availability of a relevant public dataset was scarce,there is only one relevant dataset **BCCD** and it is of substandard quality ( lower resolution and poor lighting). We take this challenge as an advantage since real world data in this domain is typically this away in several general hospitals and labs.

**Dataset :**

**BCCD** (Blood Cell Counting &amp; Detection)

This is a dataset of blood cells photos, originally open sourced by [cosmicad](https://github.com/cosmicad/dataset) and [akshaylambda](https://github.com/akshaylamba/all_CELL_data). There are 360 images across three classes: WBC (white blood cells), RBC (red blood cells), and Platelets. There are 4888 labels across 3 classes. Annotations are given in **xml** file format but our object detection models especially YOLO family takes in text format annotations as a result a wide range of preprocessing was done to feed it into these networks.

Dataset Split :

Training Set = 300 Images  Testing Set = 60 images

We noticed that there was a class imbalance. RBCs were much higher in numbers than the other two making it a more challenging task. We will discuss the difficulties faced in the forthcoming section.

After a brief discussion taking into account various pros and cons of these models we decided to start with YOLOv3 as it was a lighter model yet on par in accuracy with a heavy model which was a two stage model (Faster RCNN) and easy to change the parameters for our experiments.

**Performance on YOLOv3 on BCCD:**

Link to the dataset,trained weights,model :

[https://drive.google.com/drive/folders/1zxXjTPeQWa1ijOdbhyXZZTgXot0jNqtc?usp=sharing](https://drive.google.com/drive/folders/1zxXjTPeQWa1ijOdbhyXZZTgXot0jNqtc?usp=sharing)

Pre Processing and training details:

 1. Converted XML to .txt and recalculated the coordinates specific to YOLO architecture
 2. Calculated custom anchor boxes specific to BCCD using k- means for better and faster training
 3. Inference was done with 832 x 832 resolution as author prescribes it for getting better result
 4. Momentum = 0.9 was used with learning rate of 0.001 with decay of 0.0005
 5. Input Dimensions = 640 x 480 x 3 in batches of 64 per batch with 16 per subdivision.
 6. Total Anchors = 9 , 3 per grid cell (3 x 3)
 Recalculated anchor dimensions prior to training = 39, 38, 101, 70, 77,105, 94, 91, 114, 96, 102,109, 123,117, 172,148, 221,204
 7. Other configurations details can be found in yolo-obj.cfg file in the darknet folder in the shared drive link

With the above mentioned details the model was put to training on Google Colaboratory with GPU (Tesla K80) for a total of 6000 steps which took about 7 hrs.

Intermediate and final weights were saved to check the model performance.

![](https://user-images.githubusercontent.com/47039231/95626862-be5ced80-0a98-11eb-969f-2c8861bac6c2.png) 

A random test image was taken which the model had not seen before and each weights performance was noted. The model quickly learns to classify the components although it sometimes duplicates the counting but it gets better with steps as evident in the model predictions.

![image2](https://user-images.githubusercontent.com/47039231/95626938-e5b3ba80-0a98-11eb-99be-e7b973f97551.png) 


![image](https://user-images.githubusercontent.com/47039231/95627020-109e0e80-0a99-11eb-9248-16b48675e229.png)


The model converges to global minima in about 6 hrs and learns to differentiate the three subtypes and detects with confidence. After every 1000 steps the saved weights scores were calculated on the test set and these are as follows:

1. After 1000 weights

class\_id = 0, name = RBC, ap = 60.29% (TP = 443, FP = 157)

class\_id = 1, name = WBC, ap = 100.00% (TP = 61, FP = 0)

class\_id = 2, name = Platelets, ap = 64.46% (TP = 26, FP = 3)

for conf\_thresh = 0.25, precision = 0.77, recall = 0.58, F1-score = 0.66

for conf\_thresh = 0.25, TP = 530, FP = 160, FN = 378, average IoU = 63.75 %

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall

mean average precision (mAP@0.50) = 0.749172, or 74.92 %

2. After 2000 steps

class\_id = 0, name = RBC, ap = 87.39% (TP = 720, FP = 313)

class\_id = 1, name = WBC, ap = 94.26% (TP = 59, FP = 3)

class\_id = 2, name = Platelets, ap = 51.09% (TP = 39, FP = 40)

for conf\_thresh = 0.25, precision = 0.70, recall = 0.90, F1-score = 0.79

for conf\_thresh = 0.25, TP = 818, FP = 356, FN = 90, average IoU = 57.57 %

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall

mean average precision (mAP@0.50) = 0.775787, or 77.58 %

3. After 3000 steps

class\_id = 0, name = RBC, ap = 85.95% (TP = 698, FP = 268)

class\_id = 1, name = WBC, ap = 69.59% (TP = 47, FP = 11)

class\_id = 2, name = Platelets, ap = 88.64% (TP = 49, FP = 10)

for conf\_thresh = 0.25, precision = 0.73, recall = 0.87, F1-score = 0.80

for conf\_thresh = 0.25, TP = 794, FP = 289, FN = 114, average IoU = 55.15 %

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall

mean average precision (mAP@0.50) = 0.813893, or 81.39 %

4. After 4000 steps

class\_id = 0, name = RBC, ap = 81.47% (TP = 631, FP = 247)

class\_id = 1, name = WBC, ap = 100.00% (TP = 61, FP = 0)

class\_id = 2, name = Platelets, ap = 86.37% (TP = 40, FP = 6)

for conf\_thresh = 0.25, precision = 0.74, recall = 0.81, F1-score = 0.77

for conf\_thresh = 0.25, TP = 732, FP = 253, FN = 176, average IoU = 62.78 %

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall

mean average precision (mAP@0.50) = 0.882776, or 88.28 %

5. After 5000 steps

class\_id = 0, name = RBC, ap = 88.73% (TP = 727, FP = 288)

class\_id = 1, name = WBC, ap = 100.00% (TP = 61, FP = 0)

class\_id = 2, name = Platelets, ap = 92.44% (TP = 48, FP = 11)

for conf\_thresh = 0.25, precision = 0.74, recall = 0.92, F1-score = 0.79

for conf\_thresh = 0.25, TP = 836, FP = 299, FN = 72, average IoU = 62.64 %

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall

mean average precision (mAP@0.50) = 0.890723, or 89.07 %

6. After 6000 steps

class\_id = 0, name = RBC, ap = 86.42% (TP = 642, FP = 152)

class\_id = 1, name = WBC, ap = 100.00% (TP = 61, FP = 0)

class\_id = 2, name = Platelets, ap = 92.85% (TP = 47, FP = 4)

for conf\_thresh = 0.25, precision = 0.83, recall = 0.83, F1-score = 0.80

for conf\_thresh = 0.25, TP = 750, FP = 156, FN = 158, average IoU = 68.24 %

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall

mean average precision (mAP@0.50) = 0.900900, or 90.09 %

Loss Function and its behaviour:

The model did not learn anything for a few hundred steps so it was struggling to find a local minima however after about 300 steps loss started going down monotonically upto 600 epochs. The model started to learn and the loss gradually decreased. As evident from the loss curve we can see momentum of 0.9 surely helps the loss function from fluctuating. It took about 7hrs to reach the global minima and the training was stopped at that point.

![](https://user-images.githubusercontent.com/47039231/95627170-53f87d00-0a99-11eb-9e8f-fb2e3f34b120.png)

**Performance on Out of Bag Data:**

A google search gave an image of a blood smear. To make sure the model did not overfit to our training data or training data like conditions.

![](https://user-images.githubusercontent.com/47039231/95627258-8013fe00-0a99-11eb-837e-b48956d0fcd6.png)

The model generalizes well although it misses few RBCs which are overlapping with other WBCs but gets all the WBCs accurately.

![](https://user-images.githubusercontent.com/47039231/95627322-9de16300-0a99-11eb-8dc0-eaa4e6e3e24a.png)

**Fig : Randomly searched image**

Seeing the performance of YOLOv3, which was good but not good enough to replace any lab technician made me push the limits further. Recent Introduction of YOLOv4 gave another opportunity to try how much a single stage detector can push.

**Performance of YOLOv4 on BCCD**

Preprocessing remains the same as for YOLOv3 except the few configuration file changes which are as follows:

1. Momentum = 0.949 with learning rate of 0.001 with a decay of 0.0005
2. Batchsize =64 , subdivisions=32 as subdivisions = 16 was too much to load on the GPU .

YOLOv4 is comparatively a bulkier model (wrt YOLOv3) as a result the estimated training time was 28 hrs which is a reasonably long time for a small dataset like BCCD. However the model was trained for a limited time as Google colab has a fixed quota on GPU Usage and the results are as follows.

![](https://user-images.githubusercontent.com/47039231/95627451-e00aa480-0a99-11eb-92d3-411978902041.png)

Fig : YOLOv4 on a Blood smear

YOLOv4 performs reasonably better than YOLOv3 and the model was not even trained until 6000 steps.It was trained only for 2000 steps yet it predicts almost all the RBCs and all the WBCs with much better confidence than the best of YOLOv3, on out of bag dataset which gives us a hope that this model if trained for longer and on much better datasets can definitely achieve near to human accuracy.

![](https://user-images.githubusercontent.com/47039231/95627513-f87abf00-0a99-11eb-8835-d6045d54e66d.png)

**Fig : YOLOv4 performance on external image**

Difficulties faced while training YOLOv4

1. YOLOv4 being bulkier than its predecessor takes much longer to train
2. Loss is very unstable and fluctuates every step, this can be because of the following reasons:

1. Learning rate could be too big.
2. Large network, small dataset.
3. The model might be overfitting on the batch every step,as a result when it sees a new batch every new timestep it fails to generalize well and the loss explodes.

1. Out of all the loss curves given below only one of them tends to go down gradually.(Fig c)
2. The other two(Fig a,b) perform poorly as seen from loss curves and on the performance on out of bag images (Fig d,e) .

![](https://user-images.githubusercontent.com/47039231/95627617-1b0cd800-0a9a-11eb-99cd-172f880d69ab.png) 

Fig (a)                                                          Fig(b)

Loss curve with different initial anchor boxes and learning rates

![](https://user-images.githubusercontent.com/47039231/95627677-34ae1f80-0a9a-11eb-88ec-ce3b11d7c7c3.png)

Fig(c)

![](https://user-images.githubusercontent.com/47039231/95627752-54ddde80-0a9a-11eb-8567-cb2ddbfcf8a6.png)

Fig(d)

![](https://user-images.githubusercontent.com/47039231/95627834-7dfe6f00-0a9a-11eb-95be-f08280ff9191.png)

Fig(e)

**Future Works :**

1. YOLOv4 could be trained with a better GPU computing ability and if possible on a better dataset with good learning rate and better batch size to avoid loss fluctuation.
2. Two Stage detection networks like Faster RCNN, Mask RCNN will be implemented in the coming month similarly and compared to analyse the time vs accuracy tradeoff. So that we can take the important aspects from them and bring in some changes to it for our work. 
3. Recently Facebook AI Research published a new Object detection approach using Transformers (DeTR) which does not use anchor based and Non maximum suppression based method for training and the model achieves SOTA in many benchmarks. Implementing DeTR on blood cell detection will be a good direction to take the study forward as it shows better results than traditional object detection algorithms as it has an attention mechanism which avoids overlapping error from which traditional neural networks suffer when dealing with overcrowded objects in image, in our case RBCs. This could be a somehow incorporated into our model.
4. Similarly another model called SpineNet has been published which is current SOTA in Object Detection on MS COCO Leaderboard, implementing blood cell dataset with this model will give good insights on how to come up with a good detection model for enumerating minute entities on microscopic level in biological domain.

**References:**

In this research many papers were read and their ideas were taken and some of them were implemented and in future we hope to read and implement more such papers to come up with a more robust architecture for our use case.

1. [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
2. [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934) arXiv Apr 2020
3. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
4. [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
5. [SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization](https://arxiv.org/abs/1912.05027) 
6. [Machine Learning approach of automatic identification and counting of blood cells](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8822896)

7.[Automatic Detection and Quantification of WBCs and RBCs Using Iterative Structured Circle Detection Algorithm]       (https://www.hindawi.com/journals/cmmm/2014/979302/)

8.[Microscopic Blood smear Segmentation and classification using Deep Contour aware CNN and Extreme Machine Learning](https://ieeexplore.ieee.org/document/8014845)
