# Gender Classification (Image Classification)

Aim is to classify the gender of a person based on his/her photograph. This project assumes binary gender system for simplicity. The model was trained on my laptop's GPU (NVIDIA GTX 1650 4GB).


## Contents Of This Readme

1. [What's In The Repo](https://github.com/anantSinghCross/gender-classification/blob/master/README.md#whats-in-the-repo)
2. [Check Your Libraries](https://github.com/anantSinghCross/gender-classification/blob/master/README.md#check-your-libraries)
3. [Working of Files](https://github.com/anantSinghCross/gender-classification/blob/master/README.md#working-of-files)
4. [Dataset](https://github.com/anantSinghCross/gender-classification/blob/master/README.md#dataset)
5. [Result Snapshot](https://github.com/anantSinghCross/gender-classification/blob/master/README.md#result-snapshot)
6. [Note](https://github.com/anantSinghCross/gender-classification/blob/master/README.md#note)

## What's In The Repo

* *main.py* - This module is responsible for preparing the dataset and training the model.
* *model.h5* - This is the trained model.

## Check Your Libraries

* `Numpy`
* `Tensorflow`
* `Keras`
* `Scikit-learn`

*Instructions on how to install these libraries can be found extensively on internet.*

## Working of Files

* *main.py* - This module’s main aim is to create, prepare and train the model. Internally, also it prepares the dataset which it loads from a specific location in the machine.
Preparing the dataset includes:
   1. Extracting all the images from a specified location.
   2. Preprocessing of images which includes:
      - Converting all the images to grayscale (to reduce the processing power).
      - Resizing all the images to the same dimensions i.e. 80x110 px.
   3. Creating corresponding output values for each image from the dataset which will be used for training.

## Dataset

Dataset for training has been taken from *Kaggle*. Thanks to [Ashutosh Chauhan](https://www.kaggle.com/cashutosh) for the dataset. You can find it [here (270MB)](https://www.kaggle.com/cashutosh/gender-classification-dataset).

## Result Snapshot

![training_and_testing](/accuracyResult.PNG)

***

### Note

*More information to be added later*
