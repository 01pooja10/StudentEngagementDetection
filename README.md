# Multimodal Video Analysis: Student Engagement Detection In Online Learning

This github repository contains the implementation of the project which entails the detection of student's engagement levels in online learning.

We provide a method to aggregate three different measures of a student’s engagement levels. These measures are presented as separate modules namely, the Behavioural Traits Analysis which localizes the learner's face and detects their emotions and Facial Tracking involves micro-sleep tracking and iris distraction detection in real-time.

![Flowchart]([mitacs/dependencies/Affectation state detection.png](https://github.com/01pooja10/mitacs/blob/main/dependecies/Affectation%20state%20detection.png))

Detecting the affectation state of the learner, their frequency of micro-sleep and yawns as well as tracking their iris to detect distraction are the main goals of our proposed work. The affectation states are identified using 3D and 2D Convolutional networks trained from scratch on the DaiSEE dataset and various models are employed to study which configuration works best in terms of accuracy and computational complexity. The algorithms for facial tracking rely on 3D landmarks obtained from the student's face and further tracks certain landmarks to obtain aspect ratios.

This facilitates a comparison between spatio-temporal and solely spatial features which helps analyze the efficacy of various model architectures. The facial tracking module helps localize on the eyes and lips in order to keep track of how drowsy or distracted the students get. These modules are all combined in order to calculate an all-encompassing score which will further be used to provide real-time alerts as and when required.


