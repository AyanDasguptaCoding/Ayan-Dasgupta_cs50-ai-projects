import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy.ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    
    # Iterate through each category directory
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        
        # Iterate through each image in the category directory
        for filename in os.listdir(category_dir):
            if filename.endswith(".ppm"):  # Assuming images are in .ppm format
                image_path = os.path.join(category_dir, filename)
                
                # Read and resize the image
                image = cv2.imread(image_path)
                if image is not None:
                    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(resized_image)
                    labels.append(category)
    
    return (np.array(images), np.array(labels))


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = models.Sequential([
        # Convolutional layers with max pooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and add dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model