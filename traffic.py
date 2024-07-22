import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

KERNEL_SIZE = (3,3) # Specify kernel size for image convolution
FILTERS = 64 # Define number of filters to try using
POOL_SIZE = (3,3) # Specify pool size
DROPOUT_NEURONS = 128 # Specify no. neurons for dropout layer
DROPOUT = 0.1 # Specify percentage of neurons in layers to dropout
OUTPUT_NEURONS = 43 # Specify number of outputs (equal to no. categories)

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for i in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(i)) # concatenate dir with specific category
        
        if not os.path.isdir(category_dir): # Confirm category_dir is present
            continue # skip to next category if not

        for image_file in os.listdir(category_dir): # Iterate over each image in category
            file_path = os.path.join(category_dir, image_file) # Retrieve inidividual image paths
            
            img = cv2.imread(file_path) # Read each image as a numpy.ndarray
            res = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_AREA) # Resize image
            images.append(res) # Add image to list of images
            labels.append(int(i)) # Add image's correlating category number
    # Return completed tuples of images & labels
    return (images,labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create CNN
    model = tf.keras.models.Sequential([
        # Convolutional layer. No. of filters & kernel size specified above
        tf.keras.layers.Conv2D(
            FILTERS, KERNEL_SIZE, activation="relu", input_shape=(IMG_WIDTH,IMG_HEIGHT,3)
        ),

        # Max-pooling layer. Adjust pool-size above
        tf.keras.layers.MaxPooling2D(pool_size=POOL_SIZE),

        # Flatten units
        tf.keras.layers.Flatten(), # Pixels from resulting images fed to traditional neural network

        # Add a layer with dropout - reduce single-node dependency
        tf.keras.layers.Dense(DROPOUT_NEURONS, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(OUTPUT_NEURONS, activation="softmax")
        ])
    
    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()
