import numpy as np

import datetime

import sklearn

from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)

import os

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


def summgene(varb):
    
    print np.amin(varb)
    print np.amax(varb)
    print np.mean(varb)
    print varb.shape


def tran(indxvaluthis=None, strgvarbthis=None):
    
    listvalutemp = {}
    
    for o, strgvarb in enumerate(liststrgvarb):
        #print 'o'
        #print o
        #print 'strgvarb'
        #print strgvarb
        #print 'listvalu[strgvarb]'
        #print listvalu[strgvarb]
        #print 'numbvalu[o]'
        #print numbvalu[o]

        listvalutemp[strgvarb] = listvalu[strgvarb][numbvalu[o]/2]
    if strgvarbthis != None:
        #print 'strgvarbthis'
        #print strgvarbthis
        #print 'indxvaluthis'
        #print indxvaluthis
        listvalutemp[strgvarb] = listvalu[strgvarbthis][indxvaluthis]
    
    numbtime = listvalutemp['numbtime']
    dept = listvalutemp['dept']
    nois = listvalutemp['nois']
    
    numbdata = listvalutemp['numbdata']
    fracplan = listvalutemp['fracplan']
    numbdatabtch = listvalutemp['numbdatabtch']
    numbdimsfrst = listvalutemp['numbdimsfrst']
    numbdimsseco = listvalutemp['numbdimsseco']
    numbdatabtch = listvalutemp['numbdatabtch']
    fracdropfrst = listvalutemp['fracdropfrst']
    fracdropseco = listvalutemp['fracdropseco']

    numbdatatest = int(numbdata * fractest)
    numbdatatran = numbdata - numbdatatest
    
    numbdataplan = int(numbdata * fracplan)
    
    print 'fracplan'
    print fracplan
    print 'numbdataplan'
    print numbdataplan
    inpt = nois * np.random.randn(numbdata * numbtime).reshape((numbdata, numbtime)) + 1.
    outp = np.zeros((numbdata, 1))#random.randint(2, size=(numbdata, 1))
    
    indxinit = int(0.45 * numbtime)
    indxfinl = int(0.55 * numbtime)
    inpt[:numbdataplan, indxinit:indxfinl] *= dept
    outp[:numbdataplan, 0] = 1.
    
    indxdata = np.arange(numbdata)
    indxrand = np.random.choice(indxdata, size=numbdata, replace=False)
    inpt = inpt[indxrand, :]
    outp = outp[indxrand, :]
    
    numbdatatest = int(fractest * numbdata)
    inpttest = inpt[:numbdatatest, :]
    outptest = outp[:numbdatatest, :]
    inpttran = inpt[numbdatatest:, :]
    outptran = outp[numbdatatest:, :]


    modl = Sequential()
    modl.add(Dense(numbdimsfrst, input_dim=numbtime, activation='relu'))
    #modl.add(Dropout(fracdropfrst))
    #modl.add(Dense(numbdimsseco, activation='relu'))
    #modl.add(Dropout(fracdropseco))
    modl.add(Dense(1, activation='sigmoid'))
    
    #print modl.summary()
    modl.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[as_keras_metric(tf.metrics.precision), 'accuracy', as_keras_metric(tf.metrics.recall)])
   
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
            
            trne = matrconf[0, 0]
            flpo = matrconf[0, 1]
            flne = matrconf[1, 0]
            trpo = matrconf[1, 1]
            
            metr[y, r, 0] = trpo / float(trpo + flpo)
            metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne)
            metr[y, r, 2] = trpo / float(trpo + flne)
            
            print ' metr[y, r, 0]'
            print  metr[y, r, 0]
            print 
            #print 'outp, outppred'
            #for k in range(numbdatatemp):
            #    print outp[k, :], outppred[k, :]
            #print 'np.sum(outp)'
            #print np.sum(outp)
            #print 'np.sum(outppred)'
            #print np.sum(outppred)
            #print 'score'
            #print score
            #print 'outppred'
            #summgene(outppred)
            #print 'outp'
            #summgene(outp)
            #print 'TN FP'
            #print 'FN TP'
            #print 'I I'
            #print 'R R'
            #print 'N P'
            #print 'N P'
            #print 'matrconf'
            #print matrconf
            #print
            
    return metr

strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

pathplot = os.environ['TDGU_DATA_PATH'] + '/nnet_ssupgadn/'

fractest = 0.1

numbepoc = 30
indxepoc = np.arange(numbepoc)

numbruns = 2
indxruns = np.arange(numbruns)

listvalu = {}
listvalu['numbtime'] = np.array([3e0, 1e1, 3e1, 1e2, 3e2]).astype(int)
listvalu['dept'] = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
listvalu['nois'] = np.array([1e-3, 3e-3, 1e0, 3e-2, 1e-1])

listvalu['numbdata'] = np.array([1e2, 3e2, 1e3, 3e3, 1e4]).astype(int)
listvalu['fracplan'] = [0.1, 0.3, 0.5, 0.6, 0.9]
listvalu['numbdatabtch'] = [32, 128, 512]
listvalu['numbdimsfrst'] = [32, 64, 128]
listvalu['numbdimsseco'] = [32, 64, 128]
listvalu['fracdropfrst'] = [0.25, 0.5, 0.75]
listvalu['fracdropseco'] = [0.25, 0.5, 0.75]

liststrgvarb = listvalu.keys()
numbvarb = len(liststrgvarb) 
indxvarb = np.arange(numbvarb)

numbvalu = np.empty(numbvarb, dtype=int)
indxvalu = [[] for o in indxvarb]
for o, strgvarb in enumerate(liststrgvarb):
    numbvalu[o] = len(listvalu[strgvarb])
    indxvalu[o] = np.arange(numbvalu[o])
    print 'o'
    print o
    print 'strgvarb'
    print strgvarb
    print 'numbvalu[o]'
    print numbvalu[o]
    print

dictmetr = {}
liststrgmetr = ['prec', 'accu', 'reca']
listlablmetr = ['Precision', 'Accuracy', 'Recall']
liststrgrtyp = ['vali', 'tran']
listlablrtyp = ['Training', 'Validation']
numbrtyp = len(liststrgrtyp)
indxrtyp = np.arange(numbrtyp)

for o, strgvarb in enumerate(liststrgvarb):
    print 'strgvarb'
    print strgvarb
    print 'numbvalu[o]'
    print numbvalu[o]
    print
    dictmetr[strgvarb] = np.empty((2, 3, numbruns, numbvalu[o]))

for t in indxruns:
    
    metr = tran()
    
    #raise Exception('')

    for o, strgvarb in enumerate(liststrgvarb): 
        
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
    
        for i in indxvalu[o]:
            #print 'i'
            #print i
            #print 'strgvarb'
            #print strgvarb
            #print 'dictmetr[strgvarb]'
            #print dictmetr[strgvarb].shape
            
            metr = tran(i, strgvarb)
            
            #dictmetr[strgvarb][0, 0, t, i] = np.random.randn()
            #dictmetr[strgvarb][1, 0, t, i] = np.random.randn()
            #dictmetr[strgvarb][0, 1, t, i] = np.random.randn()
            #dictmetr[strgvarb][1, 1, t, i] = np.random.randn()
            #dictmetr[strgvarb][0, 2, t, i] = np.random.randn()
            #dictmetr[strgvarb][1, 2, t, i] = np.random.randn()
            
            dictmetr[strgvarb][0, 0, t, i] = metr[-1, 0, 0]
            dictmetr[strgvarb][1, 0, t, i] = metr[-1, 1, 0]
            dictmetr[strgvarb][0, 1, t, i] = metr[-1, 0, 1]
            dictmetr[strgvarb][1, 1, t, i] = metr[-1, 1, 1]
            dictmetr[strgvarb][0, 2, t, i] = metr[-1, 0, 2]
            dictmetr[strgvarb][1, 2, t, i] = metr[-1, 1, 2]

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

        
        axis.set_ylabel(listlablmetr[l]) 
        axis.set_xlabel(labl) 
        
        if strgvarb == 'numbdata' or strgvarb == 'numbtime' or strgvarb == 'dept' or strgvarb == 'nois':
            axis.set_xscale('log')
        plt.legend()
        plt.tight_layout()
        path = pathplot + strgvarb + strgmetr + '_' + strgtimestmp + '.pdf' 
        plt.savefig(path)
        plt.close()


