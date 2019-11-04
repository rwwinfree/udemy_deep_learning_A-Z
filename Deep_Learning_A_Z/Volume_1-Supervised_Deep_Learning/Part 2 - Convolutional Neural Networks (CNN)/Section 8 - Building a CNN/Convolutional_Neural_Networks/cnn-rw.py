
"""
Created on Wed Oct 31 15:22:24 2018

@author: ryanwinfree
"""

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN


# Importing the libraries
from keras.models import Sequential
#from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution Layer
#classifier.add(Convolution2D(32, kernel_size=(3,3), data_format='channels_last', input_shape=(64, 64, 3), activation='relu'))
classifier.add(Conv2D(32, (3,3), data_format='channels_last', input_shape=(64, 64, 3), activation='relu'))


#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a 2nd convolution layer
classifier.add(Conv2D(32, (3,3), data_format='channels_last', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Fully Connection Layer
classifier.add(Dense(units=128, activation = 'relu'))
#Using the sigmoid because we have a binary outcome (cats or dogs)
#This is the layer that gives us the final result. So we get 1 result, cat or dog
#Thus we have 1 unit (layer)
classifier.add(Dense(units=1, activation = 'sigmoid'))

#Compiling the CNN
#using the binary_crossentropy due to our classificaion and because we have a binary outcome
#If more than binary then we would need to use categorical crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

##############################################################
#Part 2 - Fitting the CNN to our images
from keras.preprocessing.image import ImageDataGenerator

#pixals take value between 1-255 so by rescaling by 1/255 it will make all our 
#rescaled values between 0-1
#shear range = applying random transfomrations
#zoom range = applies random zooms
#Horizontal flip will apply a flip to some images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = test_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

classifier.fit_generator(train_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data=validation_set,
        validation_steps=200)


# Homework Assignment
import numpy as np
from keras.preprocessing import image

#Loading the specific image we want to predict
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64,64)) 
#must now convert the image to an array to match the format we are passing into the 1st layer (input_shape)
test_image = image.img_to_array(test_image)

#Must add a batch dimension (4th dimension) because the predict method requires 4 dimensions even
#if we are using a single input...confusing
test_image = np.expand_dims(test_image, axis = 0)

#Run the model to predict on the test image
result = classifier.predict(test_image)

#Now need to map the result to the corresponding values for cats and dogs
train_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
   prediction = 'cat'     


