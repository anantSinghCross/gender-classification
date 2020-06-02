# Gender Classification (Image Classification)

Aim is to classify the gender of a person based on his/her photograph. This project assumes binary gender system for simplicity. The model was trained on my laptop's GPU (NVIDIA GTX 1650 4GB).

## What's In The Repo

* *main.py* - This module is responsible for preparing the dataset and training the model.
* *model.h5* - This is the trained model.

## Check Your Libraries

* `Numpy`
* `Tensorflow`
* `Keras`
* `Scikit-learn`

*Instructions on how to install these libraries can be found extensively on internet.*

## Working of Files in *Real-time Files* Folder

* *main.py* - This module’s main aim is to create, prepare and train the model. Internally, also it prepares the dataset which it loads from a specific location in the machine.
Preparing the dataset includes:
   1. Extracting all the images from a specified location.
   2. Preprocessing of images which includes:
      - Converting all the images to grayscale (to reduce the processing power).
      - Resizing all the images to the same dimensions i.e. 80x110 px.
   3. Creating corresponding output values for each image from the dataset which will be used for training.

## Dataset

Dataset for training has been taken from *Kaggle*. Thanks to [Ashutosh Chauhan](https://www.kaggle.com/cashutosh) for the dataset. You can find it [here](https://www.kaggle.com/cashutosh/gender-classification-dataset).

## Result Snapshot

![training_and_testing](/accuracyResult.PNG)

***

*More information to be added later*
