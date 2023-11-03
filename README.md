# Action-Recognition
## 3D CNN action Recognition using pytorch lightning and Pytorch Video API on UCF101 dataset!
### In this Project
* I utilize Pytorch Lightning library to showcase quick deep learning framework for Video analysis
* As well as showcase Pytorch Video API and its useful functionality to automate video pre-processing and dataset managing steps

X3D-M model [1] is used to showcase effectiveness of a light model. X3D models are efficient family of models raging from xs to xxl that progressively expands across the SpatioTemporal dimensions, network depth and operation channels, as well as the frame rate. Second set of experiments were conducted using SlowFast model [2] which uses a late fusion approach utilising two pathway architectures controlling the framerate. With slow pathway using a slow frame rate and fast pathway using the fast pathway.

### Data Exploration 
* Firstly here is how the dataset splits are across all of the different experimentations inspired by [@GuyKabiri](https://github.com/GuyKabiri/Video-Classification)
![](https://github.com/Ronnn007/Action-Recognition/blob/main/Graph/Data%20exploration.png)

### Data Samples
* Here are some of the examples of the frames used for training. Within the expriments range of frames are used during model training. 3 classes are selected Below:
  * Soccer Penalty ![](https://github.com/Ronnn007/Action-Recognition/blob/main/Graph/penalty%20frames.jpg)
  * Bowling ![](https://github.com/Ronnn007/Action-Recognition/blob/main/Graph/Bowling%20frames.jpg)
  * Playing Flute ![](https://github.com/Ronnn007/Action-Recognition/blob/main/Graph/Flute%20frames.jpg)

### 3D Convolution Operations
* Briefly the 3d convolution operation is showcased below:
  ![](https://github.com/Ronnn007/Action-Recognition/blob/main/Graph/3D%20Convolution%20operation.jpg)
### Training
* For Training Dynamic frames are used, rather than using a static batch where frames are extracted to jpg and than used for training. Instead Dynamic frames help address overfitting by selecting different set of frames for videos. Addtionally, Temporal subsample is also expremented with.
  
* First Model X3D-M: The illustration showcases the operations gradually expanding over the course of model training
  
![](https://github.com/Ronnn007/Action-Recognition/blob/main/Graph/X3D%20MODEL.jpg)

* Slowfast Model: Finally the second model is showcased where both pathways are fused by lateral connections as they share information. The slow pathway can be a backbone of any convolution model as it utilises larger temporal stride and processes only one out of the several temporal frames. In contract to this, the fastpathway works with a small temporal stride that is calculated by a frame rate ratio of two pathways.
  
![](https://github.com/Ronnn007/Action-Recognition/blob/main/Graph/Slowfast%20model.jpg)

Unilatral Training parameters accross each expriments For X3D M were:
* All models were pre-trained on Knietcs-400 dataset (PytorchVideo Hub)
* Dynamic frames were used throughout
* Frames in the range of 6, 8 and 16 are tested
* Optimizer: ASGD
* Scheduler: StepLR (factor= 0.1, patience=10)
* Max_epochs = 20
* Early stopping: monitoring Validation accuracy for (patience=5)
* Batch_size: 8
* Learning Rate: 1e-2
  
### Results
(Waiting to upload)
