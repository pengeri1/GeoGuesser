# GeoGuesser

## About the Project
Accurate geolocation is pivotal in urban planning, law enforcement, and technology applications.
The goal of the project is simple and easy to follow. Given a street view image from one of the 4
cities, determine which city it is from. The method ensures that the images from random
coordinates will have variability and minimal location bias for optimal training efficiency. This
project is interesting as research shows that there are many ways of tackling similar problems, which
gives us a lot of room to experiment and find a viable model for our project. By leveraging deep
learning, we aim to pair cities with street view images, enabling the ability to extract important information
from large amounts of data sets.

Why deep learning? Well many cities have small details that differ from others, these can be hard to
see and make note of, which is why deep learning is perfect. Given deep learning’s improvement towards identifying obscure patterns within images for tasks ranging from facial recognition to object
identification, we believe that we can find a model capable of accurately identifying cities based on
their street view images.

## Baseline Model
Used a pretrained AlexNet model for feature extraction and a simple two-layered
ANN to categorize images into the four cities as the baseline model. By using transfer learning, the
model would require minimal training (only the fully connected layers would be trained).
The baseline model used the pretrained AlexNet model from torchvision. For categorization, two
fully connected layers were used to downsample from 9216 to four. Cross entropy loss and stochastic
gradient descent were used to update weights in the fully connected layers.

The custom classification layers consist of:

Fully Connected Layer (fc1): This layer has 9216 input features (corresponding to the output of the
last convolutional layer in AlexNet) and 100 output features

Fully Connected Layer (fc2): This layer has 100 input features and 4 output features, corresponding
to the four cities.

The model is trained using the cross-entropy loss function and stochastic gradient descent (SGD)
optimizer to update the weights in the fully connected layers. The batch size is set to 64, and the
learning rate is initialized to 0.001. The training is conducted over a specified number of epochs.

The final training accuracy achieved by the baseline model is 0.90, while the validation accuracy is
0.55.

## My Model

The model is based on the pytorch version of the Places205 GoogleNet caffe
model from the MIT Computer Science and Artificial Intelligence Laboratory. The
structure of the model is presented as follows:

![Alt text](images/Screenshot%20(838).png)

## My Model Training

Performed in the following order:
1. Fine-tuning the GoogleNet Places205 model

   (a) Freeze the weights of all inception blocks up to Inception 5b

    (b) Disconnect Inception 5b from the GoogleNet model and attach it to the front of a custom 2-layer fully connected classifier.
  
    (c) Using the GoogleNet minus Inception 5b model, perform feature extraction on training data.
  
    (d) Train the model from (b), using features extracted from (d).
  
    (e) After training, disconnect Inception 5b from the custom classifier and reattach it to the end of the GoogleNet minus Inception 5b model to get a fine-tuned GoogleNet model.
  
3. Training the SVM

    (a) Perform feature extraction on the training dataset using the fine-tuned GoogleNet model.

    (b) Using the extracted features, train the SVM using grid-search to obtain the best set of SVM hyperparameters.

## Performance
![Alt text](images/Screenshot%20(840).png)
![Alt text](images/Screenshot%20(841).png)
![Alt text](images/Screenshot%20(842).png)
