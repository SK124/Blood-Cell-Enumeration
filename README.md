# Blood-Cell-Enumeration

**Abstract** 

Blood Cell consists of RBCs, WBCs, Platelets and their respective subtypes, everytime doctors need a report, lab technicians take hours to count and estimate and the report takes a day or two to arrive in some cases. Recent developments in Object detection algorithms have paved the way to automate many computer vision tasks detection in particular such as vehicle detection, which would not have been possible with traditional convolutional networks. This concept could be used in biological discipline where automation is scarce and in dire need. We are planning to build a novel object detection architecture specific for identifying and accurately enumerating minute and densely packed entities on microscopic level. 

**Approach**

We started with the standard object detection algorithms such as Faster RCNN, YOLOv3, SSD. We understood their architecture and implemented them on autonomous driving datasets like KITTI, BDD100k and understood their pros and cons such as ease of preprocessing, accuracy, speed and complexity of the network. Our implementation on these datasets can be found here. We understood that our use case demands more accuracy and could trade off  speed for better precision. 

In the next step we decided to implement these models on the Blood Cell Dataset and train them from scratch and compare their performances. Availability of a relevant public dataset was scarce,there is only one relevant dataset BCCD and it is of substandard quality ( lower resolution and poor lighting). We take this challenge as an advantage since real world data in this domain is typically this away in several general hospitals and  labs.

**Dataset**  
BCCD (Blood Cell Counting & Detection)

This is a dataset of blood cells photos, originally open sourced by cosmicad and  akshaylambda.    
There are 360 images across three classes: WBC (white blood cells), RBC (red blood cells), and Platelets. There are 4888 labels across 3 classes. Annotations are given in xml file format but our object detection models especially YOLO family takes in text format annotations as a result a wide range of preprocessing was done to feed it into these networks.

Dataset Split :
> Training Set = 300 Images                              Testing Set = 60 images

We noticed that there was a class imbalance. RBCs were much higher in numbers than the other two making it a more challenging task. We will discuss the difficulties faced in the forthcoming section.

After a brief discussion taking into account various pros and cons of these models we decided to start with YOLOv3 as it was a lighter model yet on par in accuracy with a heavy model which was a two stage model (Faster RCNN) and easy to change the parameters for our experiments.

**Performance on YOLOv3 on BCCD**:

Pre Processing and training details:

  1. Converted XML to .txt and recalculated the coordinates specific to YOLO architecture

  2. Calculated custom anchor boxes specific to BCCD using k- means for better and faster training

  3. Inference was done with 832 x 832 resolution as author prescribes it for getting better result

  4. Momentum = 0.9 was used with learning rate of 0.001 with decay of 0.0005 

  5. Input Dimensions = 640 x 480 x 3 in batches of 64 per batch with 16 per subdivision. 

  6. Total Anchors = 9 , 3 per grid cell (3 x 3) 

  7. Recalculated anchor dimensions prior to training = 39, 38, 101, 70,  77,105,  94, 91, 114, 96, 102,109, 123,117, 172,148, 221,204

  8. Other configurations details can be found in yolo-obj.cfg file in the darknet folder in the shared drive link 

With the above mentioned details the model was put to training on Google Colaboratory with GPU (Tesla K80) for a total of 6000 steps which took about 7 hrs. 
Intermediate and final weights were saved to check the model performance.

![2k iters](https://user-images.githubusercontent.com/47039231/95623776-75ef0100-0a93-11eb-8ab4-4c81387efe14.png)

A random test image was taken which the model had not seen before and each weights performance was noted. The model quickly learns to classify the components although it sometimes duplicates the counting but it gets better with steps as evident in the model predictions.

![image](https://user-images.githubusercontent.com/47039231/95624797-0f6ae280-0a95-11eb-9cc7-db0a9cad799d.png)



