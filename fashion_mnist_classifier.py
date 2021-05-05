# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:22:44 2021

@author: shrey

Fashion_Mnist_Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/130KtFF5H6DQ5vDDGY-QW3-PzJUGR9UKr
    
Resnet Implementation using CNNs based on past project, original implementation can be found here:
    https://github.com/bshreyas13/Deep_Learning
"""

## Import Libraries in this block ##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse as ap
import sys

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model
from keras.utils import plot_model
from keras.utils import to_categorical
import numpy as np
import os

from keras.applications import ResNet50 ## Needed for pretrained network
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as scorer
import matplotlib.pyplot as plt



## Funtion to load and normalize data ##

def Load_Datset(dataset):
  if dataset == "fashion_mnist":
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  print("Train Data shape for {}:{}{}".format(dataset,x_train.shape, y_train.shape))
  print("Test Data shape for {}:{}{}".format(dataset,x_test.shape, y_test.shape))

  return x_train, x_test,y_train,y_test

##Learning Rate Schedule ##
def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def Train_Test_Plot(model, x_train,y_train,x_val,y_val,x_test,y_test, optimizer,epochs,batch_size):

  # Compile model
  model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
  
  # prepare model model saving directory.
  save_dir = os.path.join(os.getcwd(), 'divercity_models')
  model_name = 'resnet20_model.{epoch:03d}.h5' 
  if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
  filepath = os.path.join(save_dir, model_name)

  # prepare callbacks for model saving and for learning rate adjustment. 
  steps_per_epoch=len(x_train)//batch_size
  save_period = 20
  checkpoint = ModelCheckpoint(filepath=filepath,
                               monitor='val_acc',
                               verbose=1,
                               save_freq=int(steps_per_epoch*save_period))

  lr_scheduler = LearningRateScheduler(lr_schedule)

  lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

  callbacks = [checkpoint, lr_reducer, lr_scheduler]
  
  #augmented data
  datagen = ImageDataGenerator(
      # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=20,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=True
      )
  datagen.fit(x_train)
  # Train the model 
  history= model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_val,y_val),
                        epochs=epochs, verbose=1, workers=4,
                        steps_per_epoch=len(x_train)//batch_size,
                        callbacks=callbacks)
 
  # Evaluate Model on Test set
  score = model.evaluate(x_test,
                       y_test,
                       batch_size=batch_size,
                       verbose=2)
  print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
  
  #Plot training curve
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
  
  
def Evaluate_Pred(model,x_test,y_test):

  y_pred = model.predict(x_test)
  y_pred = np.argmax(y_pred, axis=1)
  y_test = np.argmax(y_test, axis=1)
  cm = confusion_matrix(y_test,y_pred)
  scores= scorer(y_test,y_pred)

  return cm, scores

## Funtion to build Resnet ##
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
  
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def Resnet_v1(input_shape, depth, num_classes):

    # Start model definition.
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    #x = AveragePooling(pool_size=8)(x)
    y = Flatten()(x)
    x = AveragePooling2D(pool_size=4)(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-e","--epochs", required = False, help = "number of epochs to train")
    parser.add_argument("-bs","--batch_size", required = False, help = "batch_size for training")
    parser.add_argument("-tmo","--test_model_only",action='store_true',required = False, help = "Flag for choosing testing only")
    parser.add_argument("-mp","--model_path", required = False, help = "path to trained model")
    args = vars(parser.parse_args())
    
    test_only = args['test_model_only']
    model_path = args['model_path']

    #Load dataset
    x_train, x_test,y_train,y_test = Load_Datset("fashion_mnist")
    input_shape = ( x_train.shape[1], x_train.shape[2],1)
    
    # Save x,y before reshaping/encoding to display images,labels
    x = x_train
    y = y_train
    
    # From sparse label to categorical
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print("Number of unique classes:",num_labels)
    
    ## Prepare data to fit Resnet
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=1)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], x_train.shape[2],1)
    x_test = x_test.reshape(x_test.shape[0],x_train.shape[1], x_train.shape[2],1)
    x_val = x_val.reshape(x_val.shape[0],x_train.shape[1], x_train.shape[2],1)
    
    print("Data Reshaped for Resnet")
    print(x_train.shape,x_val.shape,x_test.shape)
    
    ## Class names_from Fashion MNIST dataset
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    ## Display some examples       
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y[i]])
        plt.show()

    if test_only :
        model_loaded = tf.keras.models.load_model(model_path)
        #new_model.summary()
        loss, acc = model_loaded.evaluate(x_test, y_test, verbose=2)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
        con_mat, scores = Evaluate_Pred(model_loaded,x_test,y_test)

        print("Confusion matrix")
        print(con_mat)
        print("precision for each class:\n",scores[0])
        print("Recall for each class:\n",scores[1])
        print("F score for each class:\n",scores[2])
        sys.exit()
    
    ## Network parameters
    
    if args['batch_size'] != None and args['epochs'] != None:
        batch_size = args['batch_size']
        epochs = args['epochs']
    else:
        batch_size = 256
        epochs = 100

    n_filters =16
    ## No of residual blocks(depth=20 => n=3) ##
    n=3
    ## d = 6n+2 ##
    depth = n*6+2
    
    model=Resnet_v1(input_shape,depth,num_labels)
    
    # verify the model using graph
    # enable this if pydot can be installed
    #plot_model(model, to_file='resnet.png', show_shapes=True)
    model.summary()

    ## Training parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    

    Train_Test_Plot(model, x_train, y_train, x_val, y_val, x_test, y_test, optimizer, epochs, batch_size)

    con_mat, scores = Evaluate_Pred(model,x_test,y_test)

    print("Confusion matrix")
    print(con_mat)
    print("precision for each class:\n",scores[0])
    print("Recall for each class:\n",scores[1])
    print("F score for each class:\n",scores[2])
    
