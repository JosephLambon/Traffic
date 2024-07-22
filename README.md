# Traffic
>### CS50AI Week 5: A Convolutional Neural Network to identify which traffic sign appears in a photograph.

## Contents
1. [Project Synopsis](#project_synopsis)
2. [Project Resources](#project_resources)
3. [Setup and Usage](#setup)
4. [Demo](#video)


## <a id='project_synopsis'> Project Synopsis </a>
The aim of this coursework was to create my first Convolutional Neural Network (CNN), and depthen my understanding of the different layers within. 

Having gone over Neural Network Structures, Gradient Descent, Activation Functions, Backpropogation and Overfitting, this project was an opportunity to put this pieces together practically. Experimenting with parameters in TensorFlow helped me understand how best to optimise my CNN.

## <a id='project_resources'> Project Resources </a>
* [Numpy](https://numpy.org/)
> NumPy is an open source project that enables numerical computing with Python.

* [TensorFlow](https://www.tensorflow.org/?hl=en)
> TensorFlow is an end-to-end platform for machine learning.

## <a id='setup'> Setup and Usage </a>
#### [NOTE: Any lines of code included are intended for the command line]

### 1. Install prerequisites
a. Install [Python](https://www.python.org/) </br>
b. Install [virtualenv](https://virtualenv.pypa.io/en/latest/)
``` 
 pip install virtualenv
```
### 2. Setup virtual environment
* Create virtual environment </br>
```
# Run this line on the command line
# Replace 'env_name' with whatever your desired env's name is.

virtualenv env_name
```
* Start virtual environment
```
# This will activate your virtualenv.

source env_name/bin/activate
```
* Install required packages
```
# Running this in your command line will install all listed packages to your activated virtual environment

pip install -r requirements. txt
```
### 3. Change directory
* Change into the 'traffic' folder.

### 4. Download dataset
The CNN in this project was trained and tested on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. You should download this before proceeding.

* Data provided by J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011

### 4. Run traffic.py
```
# The usage is as below. NOTE: 'data_directory' describes the GTSRB dataset in this exmaple, and '[model.h5]' is an optional 4th parameter, defining the name of the file to which you might like to save your trained model for future use.

python traffic.py data_directory [model.h5]
```

## <a id='Example'> Demo </a>

A successul run of this python script should see command line output as such:

<img width="977" alt="Scuccess" src="https://github.com/user-attachments/assets/ac8cbdbc-8850-4345-9ffe-643f088b574b">
