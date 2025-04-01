# Fault-Diagnosis-Project

Our project aims to build and train a classifier model for the detection and diagnostic of mechanical failures on a robotic arm. Our model will work with the commands given to the arm and its deviation compared to the desired output. There will be 9 classes of failures, the first corresponding to a healthy arm, and two cases of failures for each of the four motors : "Steady state error" and "Stuck".
The training is initially done on data simulated by a digital twin before operating a transfer learning to adapt our model to a real robot.

This project will be written in Python using Pytorch.