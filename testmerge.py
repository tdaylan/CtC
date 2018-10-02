import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
import tensorflow as tf

"""
Want:
    Give locl/global
    number of layers before
    number of layers after
"""


def fcon(modl, numbdimslayr, numbtime, fracdrop=0, strgLayr='init', activation='relu'):
    """
    Functionally can be added at any point in the model

    fracdrop: fraction of drop-out
    strglayr: 'init', 'medi', 'finl'
    """

    if strgLayr == 'init':
        modl.add(Dense(numbdimslayr, input_dim=numbtime, activation='relu'))
    elif strgLayr == 'medi':
        modl.add(Dense(numbdimslayr, activation= 'relu'))
    elif strgLayr == 'finl':
        modl.add(Dense(1, activation='sigmoid'))

    if fracdrop > 0.:
        modl.add(Dropout(fracdrop))


def con1(modl, numbdimslayr, numbtime, fracdrop=0, strgLayr='init', activation='relu'):
    """
    Adds a 1D CNN layer to the network
    This should not be the last layer!

    fracdrop: fraction of drop-out
    strglayr: 'init', 'medi', 'finl'
    """

    if strgLayr == 'init':
        modl.add(Conv1D(numbdimslayr, kernel_size=numbtime, input_dim=numbtime, activation='relu'))
    elif strgLayr == 'medi':
        modl.add(Conv1D(numbdimslayr, kernel_size=numbtime, activation= 'relu'))
    
    if fracdrop > 0.:
        modl.add(Dropout(fracdrop))


def parallel(datalocl, numblayrlocl, locllayr, loclfracdrop, dataglbl, numblayrglob, glbllayr, glblfracdrop, numblayrmerg):

    locl = Sequential()
    glbl = Sequential()

    if numblocllayr > 2:
        for k in range(numblocllayr - 2):
            gdat.appdfcon(fracdrop, strglayr='medi')