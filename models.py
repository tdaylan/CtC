from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
import tensorflow as tf
import numpy as np
import keras


# FROM PAPER

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate, GlobalMaxPool1D

import tensorflow as tf

# CONVOLUTIONAL MODELS

# models from "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning"

# aka astronet
def exonet():
    loclinpt, globlinpt = 200, 2000 # hard coded for now
    padX = 'same'
    padY = 'valid'

    localinput = Input(shape=(int(loclinpt),1), dtype='float32', name='localinput') 

    x = Conv1D(kernel_size=5, filters=16, padding=padX, activation='relu', input_shape=(loclinpt,1))(localinput)
    x = Conv1D(kernel_size=5, filters=16, padding=padX, activation='relu')(x)

    x = MaxPooling1D(pool_size=7, strides=2, padding=padX)(x)

    x = Conv1D(kernel_size=5, filters=32, padding=padX, activation='relu')(x)
    x = Conv1D(kernel_size=5, filters=32, padding=padX, activation='relu')(x)

    x = MaxPooling1D(pool_size=7, strides=2, padding=padX)(x)

    # -----------------------------------------------------------------------------
    globalinput = Input(shape=(int(globlinpt),1), dtype='float32', name='globalinput')

    y = Conv1D(kernel_size=5, filters=16, padding=padY, activation='relu', input_shape=(globlinpt,1))(globalinput)
    y = Conv1D(kernel_size=5, filters=16, padding=padY, activation='relu')(y)

    y = MaxPooling1D(pool_size=5, strides=2, padding=padY)(y)

    y = Conv1D(kernel_size=5, filters=32, padding=padY, activation='relu')(y)
    y = Conv1D(kernel_size=5, filters=32, padding=padY, activation='relu')(y)

    y = MaxPooling1D(pool_size=5, strides=2, padding=padY)(y)

    y = Conv1D(kernel_size=5, filters=64, padding=padY, activation='relu')(y)
    y = Conv1D(kernel_size=5, filters=64, padding=padY, activation='relu')(y)

    y = MaxPooling1D(pool_size=5, strides=2, padding=padY)(y)

    y = Conv1D(kernel_size=5, filters=128, padding=padY, activation='relu')(y)
    y = Conv1D(kernel_size=5, filters=128, padding=padY, activation='relu')(y)

    y = MaxPooling1D(pool_size=5, strides=2, padding=padY)(y)

    y = Conv1D(kernel_size=5, filters=256, padding=padY, activation='relu')(y)
    y = Conv1D(kernel_size=5, filters=256, padding=padY, activation='relu')(y)

    y = MaxPooling1D(pool_size=5, strides=2, padding=padY)(y)
    
    """
    print('shape test:', tf.keras.backend.shape(y))
    modldummy = Model(inputs=[localinput, globalinput], outputs=[y])
    print()
    modldummy.summary()

    print('shape test:', tf.keras.backend.shape(x))
    modldummy = Model(inputs=[localinput, globalinput], outputs=[x])
    print()
    modldummy.summary()
    """

    # ------------------------------------------------------------------------------
    z = keras.layers.concatenate([x, y])
    z = Flatten()(z)
    # ------------------------------------------------------------------------------

    z = Dense(512, activation='relu')(z)
    z = Dense(512, activation='relu')(z)
    z = Dense(512, activation='relu')(z)
    z = Dense(512, activation='relu')(z)

    # ------------------------------------------------------------------------------
    
    finllayr = Dense(1, activation='sigmoid', name='finl')(z)

    # ------------------------------------------------------------------------------
    modlfinl = Model(inputs=[localinput, globalinput], outputs=[finllayr])

    modlfinl.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # modlfinl.summary()
    return modlfinl

def reduced():
    loclinpt, globlinpt = 200, 2000 # hard coded for now
    padX = 'same'
    padY = 'same'

    localinput = Input(shape=(int(loclinpt),1), dtype='float32', name='localinput') 

    x = Conv1D(kernel_size=5, filters=16, padding=padX, activation='relu', input_shape=(loclinpt,1))(localinput)

    x = MaxPooling1D(pool_size=2, strides=2, padding=padX)(x)

    x = Conv1D(kernel_size=5, filters=16, padding=padX, activation='relu')(x)

    x = GlobalMaxPool1D()(x)

    # -----------------------------------------------------------------------------
    globalinput = Input(shape=(int(globlinpt),1), dtype='float32', name='globalinput')

    y = Conv1D(kernel_size=5, filters=16, padding=padY, activation='relu', input_shape=(globlinpt,1))(globalinput)

    y = MaxPooling1D(pool_size=2, strides=2, padding=padY)(y)

    y = Conv1D(kernel_size=5, filters=16, padding=padY, activation='relu')(y)

    y = MaxPooling1D(pool_size=2, strides=2, padding=padY)(y)

    y = Conv1D(kernel_size=5, filters=32, padding=padY, activation='relu')(y)

    y = GlobalMaxPool1D()(y)


    """
    # print('shape test:', tf.keras.backend.shape(y))
    modldummy = Model(inputs=[localinput, globalinput], outputs=[y])
    print()
    modldummy.summary()

    print()

    # print('shape test:', tf.keras.backend.shape(x))
    modldummy = Model(inputs=[localinput, globalinput], outputs=[x])
    print()
    modldummy.summary()
    """
    
    # ------------------------------------------------------------------------------
    z = keras.layers.concatenate([x, y])

    
    # ------------------------------------------------------------------------------

    z = Dense(1, activation='relu')(z)

    # ------------------------------------------------------------------------------
    finllayr = Dense(1, activation='sigmoid', name='finl')(z)

    # ------------------------------------------------------------------------------
    modlfinl = Model(inputs=[localinput, globalinput], outputs=[finllayr])

    modlfinl.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return modlfinl



# -----------------------------------------------------------------


# MODULAR


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
