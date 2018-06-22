**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/training_validation_loss.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5.zip a zip file containing a trained convolution neural network 
* run1.mp4 which shows the car driving around the track
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I implemented the NVidea convolution neural network as it is described in their paper.
The model is a convolutional neural network that has a Lamda layer for normalizing the images and a cropping layer for resizing the images.
The exact architecture is shown in shown later in the writup under bullet point `Final Model Architecture`

#### 2. Attempts to reduce overfitting in the model

There was no big difference between the training loss and validation loss. Therefore my model did not seem to overfit.
I did however experiment with dropout layers to see what would happen. I implemented them at different stages in the network with different settings.
This made my training and validation loss worse, so they did not end up in the final model.

Following is an image of the training and validation loss:

![alt text][image1]

#### 3. Model parameter tuning

The model uses the Adam optimizer for which I tried different learning rate settings. The different setting did not make a big difference, so I settled on the default one (0.001)
I also experimented with different epoch and sample sizes. I had good results with the following setting although differences where small.

Parameters:

* Optimizer: Adam
* Learning rate: 0.001
* Number of epochs: 5
* Sample size: 6428

#### 4. Appropriate training data

I used the training data that was supplied by Udacity.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was setting up a very simple shallow neural network to see how it performce and have a benchmark.
After that I implemented the NVIdea network. This gave much better performance, but did not let the car drive around the track.

The car seemed to have a bias steering to the left.
Generating extra training data, that flipped images horizontal, fixed the left steering bias and let the car drive around the track successfully.

#### 2. Final Model Architecture

The final model architecture (model.py lines 79-131) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Lambda                | output 160x320x3 RGB image                    | 
| Cropping              | output 90x320x3 RGB image                     | 
| Convolution 5x5       | 2x2 stride, valid padding, outputs 43x158x24  |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 20x77x36   |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 8x37x48    |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 6x35x64    |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 4x33x64    |
| Flatten               | outputs 8488                                  |
| Dense                 | output 1164                                   |
| Dense                 | output 100                                    |
| Dense                 | output 50                                     |
| Dense                 | output 10                                     |
| Dense                 | output 1                                      |

#### 3. Creation of the Training Set & Training Process

I used the training data that was supplied by Udacity, since I found it difficult to drive the car with my keyboard.
I added extra data to prevent the model to have a bias to one steering direction. This is done by randomly flipping images horizontal where the steering direction is bigger than 0.1 (model.py lines 65-67)

This was enough to have decent training and validation loss, and drive the car around the track

