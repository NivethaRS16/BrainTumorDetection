import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
  
#Training model
model = Sequential()   ## creating a blank model
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))    ### reduce the overfitting

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())    ### input layer
model.add(Dense(64,activation='relu'))    ## hidden layer of ann
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))   ## output layer

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

#Moulding train images
train_datagen = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)

test_dataset = image.ImageDataGenerator(rescale=1./255)

#Reshaping test and validation images 
train_generator = train_datagen.flow_from_directory(
    'dataset/Train',
    target_size = (224,224),
    batch_size = 10,
    class_mode = 'binary')
validation_generator = test_dataset.flow_from_directory(
    'dataset/Val',
    target_size = (224,224),
    batch_size = 10,
    class_mode = 'binary')

#### Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=7,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps=1
)

model.save("Brain tumor training.h5")
