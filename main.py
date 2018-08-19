import numpy as np

import datetime, os

from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

import sklearn
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)

def summgene(varb):
    '''
    convenience function to quickly print a numpy array
    '''
    
    print np.amin(varb)
    print np.amax(varb)
    print np.mean(varb)
    print varb.shape


def tran(indxvaluthis=None, strgvarbthis=None):
   
    '''
    Function to train the model
    '''
    
    # list of values for the variable -- by default takes the central values defined in listvalu
    listvalutemp = {}
    for o, strgvarb in enumerate(liststrgvarb):
        listvalutemp[strgvarb] = listvalu[strgvarb][numbvalu[o]/2]
    
    if strgvarbthis != None:
        # for the variable of interest, strgvarbthis, take the relevant value indexed by indxvaluthis
        listvalutemp[strgvarbthis] = listvalu[strgvarbthis][indxvaluthis]
    
    # number of time bins
    numbtime = listvalutemp['numbtime']
    # transit depth
    dept = listvalutemp['dept']
    # standard deviation of noise
    nois = listvalutemp['nois']
    
    # number of data samples
    numbdata = listvalutemp['numbdata']
    # fraction of signal in the data set
    fracplan = listvalutemp['fracplan']
    # number of data samples in a batch
    numbdatabtch = listvalutemp['numbdatabtch']
    # number of dimensions of the first fully-connected layer
    numbdimsfrst = listvalutemp['numbdimsfrst']
    # number of dimensions of the second fully-connected layer
    numbdimsseco = listvalutemp['numbdimsseco']
    # fraction of nodes to be dropped-out in the first fully-connected layer
    fracdropfrst = listvalutemp['fracdropfrst']
    # fraction of nodes to be dropped-out in the second fully-connected layer
    fracdropseco = listvalutemp['fracdropseco']
    
    # number of test data samples
    numbdatatest = int(numbdata * fractest)
    # number of training data samples
    numbdatatran = numbdata - numbdatatest
    # number of signal data samples
    numbdataplan = int(numbdata * fracplan)
    
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
    
    # randomize the data set
    indxdata = np.arange(numbdata)
    indxrand = np.random.choice(indxdata, size=numbdata, replace=False)
    inpt = inpt[indxrand, :]
    outp = outp[indxrand, :]
    
    # divide the data set into training and test data sets
    numbdatatest = int(fractest * numbdata)
    inpttest = inpt[:numbdatatest, :]
    outptest = outp[:numbdatatest, :]
    inpttran = inpt[numbdatatest:, :]
    outptran = outp[numbdatatest:, :]

    # construct Keras model
    modl = Sequential()
    modl.add(Dense(numbdimsfrst, input_dim=numbtime, activation='relu'))
    modl.add(Dropout(fracdropfrst))
    modl.add(Dense(numbdimsseco, activation='relu'))
    modl.add(Dropout(fracdropseco))
    modl.add(Dense(1, activation='sigmoid'))
    modl.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
   
    metr = np.empty((numbepoc, 2, 3))
    for y in indxepoc:
        hist = modl.fit(inpt, outp, epochs=1, batch_size=numbdatabtch, validation_split=fractest, verbose=2)
        
        for r in indxrtyp:
            if r == 0:
                inpt = inpttran
                outp = outptran
                numbdatatemp = numbdatatran
            else:
                inpt = inpttest
                outp = outptest
                numbdatatemp = numbdatatest
            
            outppred = (modl.predict(inpt) > 0.5).astype(int) 
            score = modl.evaluate(inpt, outp, verbose=0)
            matrconf = confusion_matrix(outp[:, 0], outppred[:, 0])
           
            # below is the structure of the confusion matrix for reference
            # P: Positives
            # N: Negatives
            # I: Irrelevant
            # R: Relevant

            #print 'TN FP'
            #print 'FN TP'
            
            #print 'I I'
            #print 'R R'
            
            #print 'N P'
            #print 'N P'
            
            # true negatives, false positives, false negatives and true positives
            trne = matrconf[0, 0]
            flpo = matrconf[0, 1]
            flne = matrconf[1, 0]
            trpo = matrconf[1, 1]
            
            # calculate the metrics
            metr[y, r, 0] = trpo / float(trpo + flpo)
            metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne)
            metr[y, r, 2] = trpo / float(trpo + flne)
            
    return metr

# time stamp string
strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# path where plots will be generated
pathplot = os.environ['TDGU_DATA_PATH'] + '/nnet_ssupgadn/'

# fraction of data samples that will be used to test the model
fractest = 0.1

# number of epochs
numbepoc = 30

# number of runs for each configuration in order to determine the statistical uncertainty
numbruns = 2

indxepoc = np.arange(numbepoc)
indxruns = np.arange(numbruns)

# a dictionary to hold the variable values for which the training will be repeated
listvalu = {}
# relating to the data
listvalu['numbtime'] = np.array([3e0, 1e1, 3e1, 1e2, 3e2]).astype(int)
listvalu['dept'] = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
listvalu['nois'] = np.array([1e-3, 3e-3, 1e0, 3e-2, 1e-1])
listvalu['numbdata'] = np.array([1e2, 3e2, 1e3, 3e3, 1e4]).astype(int)
listvalu['fracplan'] = [0.1, 0.3, 0.5, 0.6, 0.9]

# hyperparameters
listvalu['numbdatabtch'] = [32, 128, 512]
listvalu['numbdimsfrst'] = [32, 64, 128]
listvalu['numbdimsseco'] = [32, 64, 128]
listvalu['fracdropfrst'] = [0.25, 0.5, 0.75]
listvalu['fracdropseco'] = [0.25, 0.5, 0.75]

# list of strings holding the names of the variables
liststrgvarb = listvalu.keys()

numbvarb = len(liststrgvarb) 
indxvarb = np.arange(numbvarb)

numbvalu = np.empty(numbvarb, dtype=int)
indxvalu = [[] for o in indxvarb]
for o, strgvarb in enumerate(liststrgvarb):
    numbvalu[o] = len(listvalu[strgvarb])
    indxvalu[o] = np.arange(numbvalu[o])

# dictionary to hold the metrics resulting from the runs
dictmetr = {}
liststrgmetr = ['prec', 'accu', 'reca']
listlablmetr = ['Precision', 'Accuracy', 'Recall']
liststrgrtyp = ['vali', 'tran']
listlablrtyp = ['Training', 'Validation']
numbrtyp = len(liststrgrtyp)
indxrtyp = np.arange(numbrtyp)

for o, strgvarb in enumerate(liststrgvarb):
    dictmetr[strgvarb] = np.empty((2, 3, numbruns, numbvalu[o]))

# for each run
for t in indxruns:
    
    # do the training for the central value
    metr = tran()
    
    # for each variable
    for o, strgvarb in enumerate(liststrgvarb): 
        
        # for each value
        for i in indxvalu[o]:
           
            # do the training for the specific value of the variable of interest
            metr = tran(i, strgvarb)
            
            dictmetr[strgvarb][0, 0, t, i] = metr[-1, 0, 0]
            dictmetr[strgvarb][1, 0, t, i] = metr[-1, 1, 0]
            dictmetr[strgvarb][0, 1, t, i] = metr[-1, 0, 1]
            dictmetr[strgvarb][1, 1, t, i] = metr[-1, 1, 1]
            dictmetr[strgvarb][0, 2, t, i] = metr[-1, 0, 2]
            dictmetr[strgvarb][1, 2, t, i] = metr[-1, 1, 2]

# plot the resulting metrics
for o, strgvarb in enumerate(liststrgvarb): 
    for l, strgmetr in enumerate(liststrgmetr):
        figr, axis = plt.subplots()
        
        for r in indxrtyp:
            ydat = np.mean(dictmetr[strgvarb][r, l, :, :], axis=0)
            yerr = np.empty((2, numbvalu[o]))
            if r == 0:
                alph = 0.5
            else:
                alph = 1.

            for i in indxvalu[o]:
                yerr[0, i] = ydat[i] - np.percentile(dictmetr[strgvarb][r, l, :, i], 5.)
                yerr[1, i] = np.percentile(dictmetr[strgvarb][r, l, :, i], 95.) - ydat[i]
            
            if r == 0:
                linestyl = '--'
            else:
                linestyl = ''
            temp, listcaps, temp = axis.errorbar(listvalu[strgvarb], ydat, yerr=yerr, label=listlablrtyp[r], capsize=10, marker='o', ls='', markersize=10, lw=3, alpha=alph)
            for caps in listcaps:
                caps.set_markeredgewidth(3)
        
        if strgvarb == 'numbtime':
            labl = '$N_{time}$'
        
        if strgvarb == 'dept':
            labl = '$\delta$'
        
        if strgvarb == 'nois':
            labl = '$\sigma$'
        
        if strgvarb == 'numbdata':
            labl = '$N_{data}$'
        
        if strgvarb == 'fracplan':
            labl = '$f_{p}$'
        
        if strgvarb == 'numbdatabtch':
            labl = '$N_{db}$'
    
        if strgvarb == 'numbdimsfrst':
            labl = '$N_{dens,1}$'
        
        if strgvarb == 'numbdimsseco':
            labl = '$N_{dens,2}$'
    
        if strgvarb == 'fracdropfrst':
            labl = '$f_{d,1}$'
    
        if strgvarb == 'fracdropseco':
            labl = '$f_{d,2}$'
    
        axis.set_ylabel(listlablmetr[l]) 
        axis.set_xlabel(labl) 
        
        if strgvarb == 'numbdata' or strgvarb == 'numbtime' or strgvarb == 'dept' or strgvarb == 'nois':
            axis.set_xscale('log')
        plt.legend()
        plt.tight_layout()
        path = pathplot + strgvarb + strgmetr + '_' + strgtimestmp + '.pdf' 
        plt.savefig(path)
        plt.close()


