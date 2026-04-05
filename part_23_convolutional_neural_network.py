# Part 23 - Convolutional Neural Network (CNN)

# Import libraries
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import load_img, img_to_array
from keras.src.utils.image_dataset_utils import image_dataset_from_directory

print(tf.__version__)

# Data Preprocessing
# image_dataset_from_directory() replaces ImageDataGenerator() in Keras 2.10 and later
train_image_gen = image_dataset_from_directory(
    'datasets/training_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)
test_image_gen = image_dataset_from_directory(
    'datasets/test_set',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

# Building the Sequential model
model = Sequential()
# Convolution Layer
model.add(Conv2D(
    filters=32, 
    kernel_size=3, 
    activation='relu', 
    input_shape=[64, 64, 3]
    )
)
# Pooling Layer
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second Convolution and Pooling Layer
model.add(Conv2D(
    filters=32, 
    kernel_size=3, 
    activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flattening Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(units=128, activation='relu'))

# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training the model
model.fit(
    x=train_image_gen,
    validation_data=test_image_gen,
    epochs=25
)

# Converts test image to an array to be inputted into the model
# and loads test image with the same target size as the training images
# for processing
test_image = load_img('datasets/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make a single prediction
result = model.predict(test_image)
# Gets class names from the training images to determine if the image is a cat or a dog
train_image_gen.class_names
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
print("Prediction: ", prediction)