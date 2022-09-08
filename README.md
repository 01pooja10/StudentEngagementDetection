# Real-time Student Engagement Monitoring Interface

This github repository contains the implementation of the project which entails the detection of student's engagement levels in online learning using four different modalities which have all been evaluated separately and also integrated for enhancing accuracy of predictions.

We provide a method to aggregate three different measures of a student’s engagement levels. These measures are presented as separate modules namely, the Behavioural Traits Analysis which localizes the learner's face and detects their emotions while Facial Tracking involves micro-sleep tracking, yawn identification and iris distraction detection in real-time using a 3D facial mesh.

![Flowchart](https://github.com/01pooja10/StudentEngagementDetection/blob/main/dependecies/flow1re.png)

Detecting the affective state (level of engagement) of the learner, their frequency of micro-sleep and yawns as well as tracking their iris to detect distraction are the main goals of our proposed work. The affective states are identified using 3D and 2D Convolutional networks trained from scratch on the DAiSEE dataset and various models are employed to study which configuration works best in terms of accuracy and computational complexity. The algorithms for facial tracking rely on 3D landmarks obtained from the student's face and further tracks certain landmarks to obtain respective aspect ratios.

This facilitates a comparison between spatio-temporal and solely spatial features which helps analyze the efficacy of various model architectures. The facial tracking module helps localize on the eyes and lips in order to keep track of how drowsy or distracted the students get. These modules are all combined in order to calculate an all-encompassing score which will further be used to provide real-time alerts as and when required.

This system follows a sequential flow: as and when the micro-sleep tracking algorithm detects drowsiness, the pipeline is halted and otherwise, the other three modules are set in action. The neural networks for facial emotion recognition module obtain a batch of 3 frames and the last frame from this set is processed by the iris distraction and yawn identification modules. The pipeline incurs some latency due to asynchronous processing and uniform allocation of resources.

![flowchart2](https://github.com/01pooja10/StudentEngagementDetection/blob/main/dependecies/flow2.png)

Given below is a sample grid of images which depict our algorithms' real-time performance and how the instant alerts are generated on screen while the user (student) is undergoing learning process in an online environment.

![grid](https://github.com/01pooja10/mitacs/blob/main/dependecies/gridim.png)

## Modules used

1. **Facial Emotion Detection** - We make use of 2D convolutions and 3D convolutions thereby facilitating the involvement of a comparison between frame-by-frame sequential image classification and 3D (temporal) frame processing by passing both lone frames (1 frame) and a continuous sequence of frames as inputs. Hence, we construct a myriad of models which accept both spatial and temporal data. We also include the VGG-16 network architecture for comparing its performance with other residual networks of various depths.
2. **Micro-sleep Tracking** - Whenever students display signs of sleepiness or drowsiness, our algorithm for detecting micro-sleep patterns, observes and accurately tracks a deviation from the engaged state. This helps add an extra precautionary measure to ensuring that students stay actively focused throughout the lectures being taught online.
3. **Yawn Identification** - Yawns are also passive indicators of lack of concentration and a decreasing attention span. So, we propose an algorithm which tracks the lips in real-time and calculates the aspect ratio by singling out the coordinates pertaining to the mid and end points of the upper and lower lips.
4. **Iris Distraction Detection** - We propose the usage of a real-time iris monitoring algorithm for understanding the more nuanced aspects of how well a student is engaged during a lecture. The iris is usually centered at the camera when a concentrated student is involved in the online session.


## Dataset used
We have employed the DAiSEE dataset for training 2D and 3D convolutional neural networks as it had a myriad of settings and varied facial expressions which were categorised under appropriate classes. We further consolidated the labels into binary values by using an algorithm which measured how engaged the learners were while assigning equal importance to other labels: confusion, frustration and boredom.

![daiseegrid](https://github.com/01pooja10/StudentEngagementDetection/blob/main/dependecies/grid2.png)

Manually labeled images obtained from a locally sourced webcam were also employed to test the micro-sleep tracking and iris distraction detection modules to ensure proper performance. Around 100 images were used to test each algorithm to measure its real-time performance.

![manual](https://github.com/01pooja10/StudentEngagementDetection/blob/main/dependecies/all3c.png)

## Installation and Quick Start
To use the repo and run real-time inference on your local machine using a valid camera source, please follow the steps given below:

- Clone, enter, set up and run the files in the repository: 

        $ git clone https://github.com/01pooja10/StudentEngagementDetection
        $ cd StudentEngagementDetection/
        $ pip install -r requirements.txt
        $ python run.py
        
- The command below can be executed for integrating all facial tracking modules with the 3D ResNet model predictions on 3 consecutive frames:

        $ python runmod.py
        
- The command below can be executed for selecting convolutional models to be trained on custom/DAiSEE dataset:

        $ python train.py  

This repository also contains the code files used to construct each module and the entire notebook used to train models on the DAiSEE dataset using a NVIDIA GPU and cuda on the PyTorch framework. The notebook can be accessed by downloading it and running it on a local GPU after ensuring that the command given below is successfully executed.

        $ torch.cuda.get_device_name(0)
        
The different Python libraries used include:
* PyTorch
* OpenCV
* Pillow
* Scipy
* Numpy
* Matplotlib
* Mediapipe

These libraries along with their versions are listed in the requirements.txt file for reference purposes. Tjis repository can be run in 2 ways for real-time inference purposes and a training notebook attached provides more insight into how the various models were trained on the preprocessed image dataloaders.
