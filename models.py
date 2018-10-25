from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
import tensorflow as tf
import numpy as np
import keras

# currently can't call this in the test function!
def twoinput(dataclass, layers, fracdropbool=True):
    """
    dataclass is an instance of gdat

    layers is a list of layer quantities:
        [0] := number of local
        [1] := number of global
        [2] := number after being combined 
    """
    numbtime = dataclass.numbtime
    numbdimslayr = dataclass.numbdimslayr
    fracdrop = dataclass.fracdrop
    
    # ----------------------------------------------------------------------------
    localinput = Input(shape=(int(numbtime),), dtype='float32', name='localinput')

    x = Dense(numbdimslayr, input_dim=numbtime, activation='relu')(localinput)
    
    if fracdropbool:
        x = Dropout(fracdrop)(x)

    for i in range(layers[0]-1):
        x = Dense(numbdimslayr, activation='relu')(x)
        
        if fracdropbool:
            x = Dropout(fracdrop)(x)


    # -----------------------------------------------------------------------------
    globalinput = Input(shape=(int(numbtime),), dtype='float32', name='globalinput')

    y = Dense(numbdimslayr, input_dim=numbtime, activation='relu')(globalinput)
    
    if fracdropbool:
        y = Dropout(fracdrop)(y)

    for i in range(layers[1]-1):
        y = Dense(numbdimslayr, activation='relu')(y)
        
        if fracdropbool:
            y = Dropout(fracdrop)(y)

    # ------------------------------------------------------------------------------
    z = keras.layers.concatenate([x, y])

    # ------------------------------------------------------------------------------
    if layers[2] >=2:
        z = Dense(numbdimslayr, activation='relu')(z)
        
        for i in range(layers[2]-2):
            z = Dense(numbdimslayr, activation='relu')(z)

    # ------------------------------------------------------------------------------
    finllayr = Dense(1, activation='sigmoid', name='finl')(z)

    # ------------------------------------------------------------------------------
    modlfinl = Model(inputs=[localinput, globalinput], outputs=[finllayr])

    modlfinl.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return modlfinl


# this one works
def singleinput(dataclass,fracdropbool=True):
    """
    dataclass : instance of gdat

    layers : number of layers 

    fracdropbool : true or false on doing the fracdrop
    """

    numbtime = dataclass.numbtime
    numbdimslayr = dataclass.numbdimslayr
    fracdrop = dataclass.fracdrop
    layers = dataclass.numblayr
    inptshape = dataclass.inpt.shape
    unused, useddim = inptshape
    

    input_S = Input(shape=(useddim,), dtype='float32', name='input')

    x = Dense(numbdimslayr, input_dim=numbtime, activation='relu')(input_S)
    
    if fracdropbool:
        x = Dropout(fracdrop)(x)

    for i in range(layers-1):
        x = Dense(numbdimslayr, activation='relu')(x)
        
        if fracdropbool:
            x = Dropout(fracdrop)(x)

    finllayr = Dense(1, activation='sigmoid', name='finl')(x)
    
    modlfinl = Model(inputs=[input_S], outputs=[finllayr])

    modlfinl.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return modlfinl
