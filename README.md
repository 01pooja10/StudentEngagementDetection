# Multimodal Video Analysis: Student Engagement Detection In Online Learning

This github repository contains the implementation of the project which entails the detection of student's engagement levels in online learning using three different modalities which have all been evaluated separately and also integrated for enhancing accuracy of predictions.

We provide a method to aggregate three different measures of a student’s engagement levels. These measures are presented as separate modules namely, the Behavioural Traits Analysis which localizes the learner's face and detects their emotions and Facial Tracking involves micro-sleep tracking and iris distraction detection in real-time.

![Flowchart](https://github.com/01pooja10/mitacs/blob/main/dependecies/Affectation%20state%20detection.png)

Detecting the affectation state of the learner, their frequency of micro-sleep and yawns as well as tracking their iris to detect distraction are the main goals of our proposed work. The affectation states are identified using 3D and 2D Convolutional networks trained from scratch on the DaiSEE dataset and various models are employed to study which configuration works best in terms of accuracy and computational complexity. The algorithms for facial tracking rely on 3D landmarks obtained from the student's face and further tracks certain landmarks to obtain aspect ratios.

This facilitates a comparison between spatio-temporal and solely spatial features which helps analyze the efficacy of various model architectures. The facial tracking module helps localize on the eyes and lips in order to keep track of how drowsy or distracted the students get. These modules are all combined in order to calculate an all-encompassing score which will further be used to provide real-time alerts as and when required.

Given below is a sample grid of images which depict our algorithms' real-time performance and how the instant alerts are generated on screen while the user (student) is undergoing learning through an online environment.

![grid](https://github.com/01pooja10/mitacs/blob/main/dependecies/gridim.png)

## Datasets used
We have employed the DAiSEE dataset for training 2D and 3D convolutional neural networks as it had a myriad of settings and varied facial expressions which were categorised under appropriate classes. We further consolidated the labels into binary values by using an algorithm which measured how engaged the learners were while assigning equal importance to other labels: confusion, frustration and boredom.

![daiseegrid](https://github.com/01pooja10/StudentEngagementDetection/blob/main/dependecies/grid2.png)

## Installation and Quick Start
To use the repo and run real-time inference on your local machine using a valid camera source, please follow the steps given below:

- Cloning the repository: 

        $ git clone https://github.com/01pooja10/StudentEngagementDetection
        
- Entering the directory: 

        $ cd StudentEngagementDetection/
        
- Setting up your Python Environment with dependencies:

        $ pip install -r requirements.txt

- Running the file for inference:

        $ python run.py
        
- The command below can be executed for integrating all facial tracking modules with the 3D ResNet model predictions on 3 consecutive frames:

        $ python runmod.py

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

These libraries along with their versions are listed in the requirements.txt file for reference purposes.
