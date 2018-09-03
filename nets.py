import numpy as np
import datetime, os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
import tensorflow as tf
import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

"""
inputs=
gdat: (class gdatstrt) the data input 
inxvaluthis: unknown 
strgarbthis: unknown

output=
metr: metrics

"""

class nets():
    def __init__(self, gdat, indxvaluthis=None, strgvarbthis=None):
        self.gdat = gdat
        listvalutemp = {}
        for o, strgvarb in enumerate(gdat.liststrgvarb):
            listvalutemp[strgvarb] = gdat.listvalu[strgvarb][gdat.numbvalu[o]/2]

        if strgvarbthis != None:
            listvalutemp[strgvarbthis] = gdat.listvalu[strgvarbthis][indxvaluthis]

        # number of time bins
        numbtime = listvalutemp['numbtime']  
        self.numbtime = numbtime
        # transit depth
        dept = listvalutemp['dept']
        # standard deviation of noise
        nois = listvalutemp['nois']
        
        # number of data samples
        numbdata = listvalutemp['numbdata']
        # fraction of signal in the data set
        fracplan = listvalutemp['fracplan']
        # number of data samples in a batch
        self.numbdatabtch = listvalutemp['numbdatabtch']
        # number of dimensions of the first fully-connected layer
        self.numbdimsfrst = listvalutemp['numbdimsfrst']
        # number of dimensions of the second fully-connected layer
        self.numbdimsseco = listvalutemp['numbdimsseco']
        # fraction of nodes to be dropped-out in the first fully-connected layer
        self.fracdropfrst = listvalutemp['fracdropfrst']
        # fraction of nodes to be dropped-out in the second fully-connected layer
        self.fracdropseco = listvalutemp['fracdropseco']
        
        # number of test data samples
        numbdatatest = int(numbdata * gdat.fractest)
        self.numbdatatest = numbdatatest
        # number of training data samples
        self.numbdatatran = numbdata - numbdatatest
        # number of signal data samples
        numbdataplan = int(numbdata * fracplan)
        
        if gdat.datatype == 'here':  
            
            # generate (background-only) light curves
            # temp -- this currently does not use the repository 'exop'
            inpt = nois * np.random.randn(numbdata * numbtime).reshape((numbdata, numbtime)) + 1.
            # set the label of all to 0 (background)
            outp = np.zeros((numbdata, 1))
            
            # time indices of the transit 
            ## beginning
            indxinit = int(0.45 * numbtime)
            ## end
            indxfinl = int(0.55 * numbtime)
    
            # lower the relevant time bins by the transit depth
            inpt[:numbdataplan, indxinit:indxfinl] *= dept
            # change the labels of these data samples to 1 (signal)
            outp[:numbdataplan, 0] = 1.
        
        if gdat.datatype == 'ete6':  
            
            exop.main.retr_ete6()

        # randomize the data set
        indxdata = np.arange(numbdata)
        indxrand = np.random.choice(indxdata, size=numbdata, replace=False)
        inpt = inpt[indxrand, :]
        outp = outp[indxrand, :]
        self.inpt = inpt
        self.outp = outp

        # divide the data set into training and test data sets
        numbdatatest = int(gdat.fractest * numbdata)
        self.inpttest = inpt[:numbdatatest, :]
        self.outptest = outp[:numbdatatest, :]
        self.inpttran = inpt[numbdatatest:, :]
        self.outptran = outp[numbdatatest:, :]

    def Sqnl(self):
        modl = Sequential()
        modl.add(Dense(self.numbdimsfrst, input_dim=self.numbtime, activation='relu'))
        modl.add(Dropout(self.fracdropfrst))
        modl.add(Dense(self.numbdimsseco, activation='relu'))
        modl.add(Dropout(self.fracdropseco))
        modl.add(Dense(1, activation='sigmoid'))
        modl.compile(loss='binary_crossentropy', optimizer='remsprop', metrics=['accuracy'])

        metr = np.zeros((self.gdat.numbepoc, 2, 3)) - 1
        loss = np.empty(self.gdat.numbepoc)
        numbepocchec = 5

        for y in self.gdat.inxepoc:
            hist = modl.fit(self.inpt, self.outp, epochs=1, batch_size=self.numbdatabtch, validation_split=self.gdat.fractest, verbose=0)
            loss[y] = hist.history['loss'][0]
            indxepocloww = max(0, y- numbepocchec)
            if y == self.gdat.numbepoc - 1 and 100. * (loss[indxepocloww] - loss[y]):
                print('Warning! The optimizer may not have converged.')
                print('loss[indxepocloww]\n', loss[indxepocloww], '\nloss[y]\n', loss[y], '\nloss\n', loss)

            for r in self.gdat.inxrtyp:
                if r==0:
                    inpt = self.inpttran
                    outp = self.outptran
                    numdatatemp = self.numbdatatran
                else:
                    inpt = self.inpttest
                    outp = self.outptest
                    numbdatatemp = self.numbdatatest

                outppred = (modl.predict(inpt) > 0.5).astype(int)
                score = modl.evaluate(inpt,outp, verpose=0)
                matrconf = confusion_matrix(outp[:, 0], outppred[:, 0])

                trne = matrconf[0, 0]
                flpo = matrconf[0, 1]
                flne = matrconf[1, 0]
                trpo = matrconf[1, 1]

                if float(trpo + flpo) > 0:
                    metr[y, r, 0] = trpo / float(trpo + flpo)
                metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne)
                if float(trpo + flne) > 0:
                    metr[y, r, 2] = trpo / float(trpo + flne)
            return metr

    
    def Cnn1D(self):
        modl = Sequential()
        modl.add(Conv1D(self.numbdimsfrst, input_dim=self.numbtime, activation='relu'))
        modl.add(Dropout(self.fracdropfrst))
        modl.add(Conv1D(self.numbdimsseco, activation='relu'))
        modl.add(Dropout(fracdropseco))
        modl.add(Dense(1, activation='sigmoid'))
        modl.compile(loss='binary_crossentropy', optimizer='remsprop', metrics=['accuracy'])

        metr = np.zeros((self.gdat.numbepoc, 2, 3)) - 1
        loss = np.empty(self.gdat.numbepoc)
        numbepocchec = 5

        for y in self.gdat.inxepoc:
            hist = modl.fit(self.inpt, self.outp, epochs=1, batch_size=self.numbdatabtch, validation_split=self.gdat.fractest, verbose=0)
            loss[y] = hist.history['loss'][0]
            inxepocloww = max(0, y- numbepocchec)
            if y == self.gdat.numbepoc - 1 and 100. * (loss[indxepocloww] - loss[y]):
                print('Warning! The optimizer may not have converged.')
                print('loss[indxepocloww]\n', loss[indxepocloww], '\nloss[y]\n', loss[y], '\nloss\n', loss)

            for r in self.gdat.inxrtyp:
                if r==0:
                    inpt = self.inpttran
                    outp = self.outptran
                    numdatatemp = self.numbdatatran
                else:
                    inpt = self.inpttest
                    outp = self.outptest
                    numbdatatemp = self.numbdatatest

                outppred = (modl.predict(inpt) > 0.5).astype(int)
                score = modl.evaluate(inpt,outp, verpose=0)
                matrconf = confusion_matrix(outp[:, 0], outppred[:, 0])



                trne = matrconf[0, 0]
                flpo = matrconf[0, 1]
                flne = matrconf[1, 0]
                trpo = matrconf[1, 1]
                

                if float(trpo + flpo) > 0:
                    metr[y, r, 0] = trpo / float(trpo + flpo)
                metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne)
                if float(trpo + flne) > 0:
                    metr[y, r, 2] = trpo / float(trpo + flne)
            return metr
