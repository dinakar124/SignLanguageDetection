# SignLanguageDetection

This repository includes an image classification model which classifies different signs used in sign language. I've included the code to train the model(Main.py) and also the code for classifying your own image(classify.py).
Download the dataset form here : https://drive.google.com/open?id=1O6GUgx2rBagG_VpyGpBhbyZc9uzajfde

Required Dependencies : Python, Tensorflow, Keras

Tensorflow and Keras Installation : https://www.tensorflow.org/install/pip (Tensorflow) https://keras.io/#installation (Keras)

First, clone the repository and download the dataset. Change the dataset path to the path where you downloaded the dataset in Train.py

Running "Main.py" trains the model and produces the keras h5 weights file (which basically contains the saved weights of the model).

"Weights_Final.h5" is the file which contains the weights of our model.

In "Classify.py", we deploy our trained Keras model to production so that we can test it on our own images.
