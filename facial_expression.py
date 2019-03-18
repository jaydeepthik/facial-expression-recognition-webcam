# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:44:07 2019

@author: jaydeep thik
"""

import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#---------------------------------------------------------------------------------------------------------------------------------
def generate_dataset():
    
    """generate dataset from csv"""
    
    df = pd.read_csv("./fer2013/fer2013.csv")
    
    train_samples = df[df['Usage']=="Training"]
    validation_samples = df[df["Usage"]=="PublicTest"]
    test_samples = df[df["Usage"]=="PrivateTest"]
    
    y_train = train_samples.emotion.astype(np.int32).values
    y_valid = validation_samples.emotion.astype(np.int32).values
    y_test = test_samples.emotion.astype(np.int32).values
     
    X_train =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in train_samples.pixels])
    X_valid =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in validation_samples.pixels])
    X_test =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape((48,48)) for image in test_samples.pixels])
    
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

#---------------------------------------------------------------------------------------------------------------------------------
    
def generate_model(lr=0.001):
    
    """training model"""
    
    with tf.device('/gpu:0'):  
        
        model = keras.models.Sequential()
        
        model.add(keras.layers.Conv2D(64,(3,3), input_shape=(48,48, 1), padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Dropout(0.20))
        
        model.add(keras.layers.Conv2D(128,(5,5), padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Dropout(0.20))

        model.add(keras.layers.Conv2D(512,(3,3), padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Dropout(0.20))
        
        model.add(keras.layers.Conv2D(512,(3,3)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Dropout(0.25))
        
        model.add(keras.layers.Conv2D(256,(3,3), activation='relu'))
        model.add(keras.layers.Conv2D(128,(3,3), padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Dropout(0.25))
        
        #model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        
        model.add(keras.layers.Dense(7,activation='softmax'))
        
        model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=lr) , metrics=['accuracy'])
        return model
    
#---------------------------------------------------------------------------------------------------------------------------------
        
if __name__=="__main__":
    
    #df = pd.read_csv("./fer2013/fer2013.csv")
    X_train, y_train, X_valid, y_valid, X_test, y_test =  generate_dataset()
    
    X_train = X_train.reshape((-1,48,48,1)).astype(np.float32)
    X_valid = X_valid.reshape((-1,48,48,1)).astype(np.float32)
    X_test = X_test.reshape((-1,48,48,1)).astype(np.float32)
    
    X_train_std = X_train/255.
    X_valid_std = X_valid/255.
    X_test_std = X_test/255.
    
    model = generate_model(0.01)
    with tf.device("/gpu:0"):
        history = model.fit(X_train_std, y_train,batch_size=128,epochs=35, validation_data=(X_valid_std, y_valid), shuffle=True)
        model.save("my_model.h5")
    
    
    
    
    
