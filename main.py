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
sns.set(context='poster', style='ticks', color_codes=True)

def summgene(varb):
    '''
    convenience function to quickly print a numpy array
    '''
    
    print (np.amin(varb))
    print (np.amax(varb))
    print (np.mean(varb))
    print (varb.shape)


def fcon(gdat, indxvaluthis=None, strgvarbthis=None):
   
    '''
    Function to train the model
    '''
    
    # list of values for the variable -- by default takes the central values defined in listvalu
    listvalutemp = {}
    for o, strgvarb in enumerate(gdat.liststrgvarb):
        listvalutemp[strgvarb] = gdat.listvalu[strgvarb][gdat.numbvalu[o]/2]
    
    if strgvarbthis != None:
        # for the variable of interest, strgvarbthis, take the relevant value indexed by indxvaluthis
        listvalutemp[strgvarbthis] = gdat.listvalu[strgvarbthis][indxvaluthis]
    
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
    numbdatatest = int(numbdata * gdat.fractest)
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
    numbdatatest = int(gdat.fractest * numbdata)
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
   
    metr = np.zeros((gdat.numbepoc, 2, 3)) - 1.
    loss = np.empty(gdat.numbepoc)
    numbepocchec = 5
    for y in gdat.indxepoc:
        hist = modl.fit(inpt, outp, epochs=1, batch_size=numbdatabtch, validation_split=gdat.fractest, verbose=0)
        loss[y] = hist.history['loss'][0]
        indxepocloww = max(0, y - numbepocchec)
        if y == gdat.numbepoc - 1 and 100. * (loss[indxepocloww] - loss[y]) / loss[y] > 1.:
            print ('Warning! The optimizer may not have converged.')
            print ('loss[indxepocloww]')
            print (loss[indxepocloww])
            print ('loss[y]')
            print (loss[y])
            print ('loss')
            print (loss)
            #raise Exception('')
        
        for r in gdat.indxrtyp:
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
            if float(trpo + flpo) > 0:
                metr[y, r, 0] = trpo / float(trpo + flpo)
            metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne)
            if float(trpo + flne) > 0:
                metr[y, r, 2] = trpo / float(trpo + flne)
            
    return metr


class gdatstrt(object):

    def __init__(self):
        pass


def expl( \
         # string indicating the model
         strguser='tansu', \
         strgtopo='fcon', \
         zoomtype='locl' # if local, operates normal, if local+globa or dub(double) it will take local and global at the same time
         datatype='here', \
        ):

    '''
    Function to explore the effect of hyper-parameters (and data properties for mock data) on binary classification metrics
    '''
    
    # global object that will hold global variables
    gdat = gdatstrt()

    # time stamp string
    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # function that will do the training for the desired topology
    functopo = globals().get(strgtopo)
    
    print ('CtC explorer initialized at %s.' % strgtimestmp)
    
    # path where plots will be generated
    pathplot = os.environ['TDGU_DATA_PATH'] + '/nnet_ssupgadn/'
    
    print ('Will generate plots in %s' % pathplot)

    # these variables are hard coded in, do we want that? probably no, make variables

    # fraction of data samples that will be used to test the model
    gdat.fractest = 0.1
    
    # number of epochs
    gdat.numbepoc = 2
    
    # number of runs for each configuration in order to determine the statistical uncertainty
    numbruns = 2
    
    # end of hard-coded vars to fix

    gdat.indxepoc = np.arange(gdat.numbepoc)
    indxruns = np.arange(numbruns)
    
    """"
    from tensorflow.python.client import device_lib
    listdictdevi = device_lib.list_local_devices()
    print ('Names of the devices detected: ')
    for dictdevi in listdictdevi:
        print (dictdevi.name)
    """

    # a dictionary to hold the variable values for which the training will be repeated
    gdat.listvalu = {}
    # relating to the data
    gdat.listvalu['numbtime'] = np.array([3e0, 1e1, 3e1, 1e2, 3e2]).astype(int)
    gdat.listvalu['dept'] = 1 - np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    gdat.listvalu['nois'] = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    gdat.listvalu['numbdata'] = np.array([1e2, 3e2, 1e3, 3e3, 1e4]).astype(int)
    gdat.listvalu['fracplan'] = [0.1, 0.3, 0.5, 0.6, 0.9]
    
    # hyperparameters
    gdat.listvalu['numbdatabtch'] = [32, 128, 512]
    gdat.listvalu['numbdimsfrst'] = [32, 64, 128]
    gdat.listvalu['numbdimsseco'] = [32, 64, 128]
    gdat.listvalu['fracdropfrst'] = [0.25, 0.5, 0.75]
    gdat.listvalu['fracdropseco'] = [0.25, 0.5, 0.75]
    
    # list of strings holding the names of the variables
    gdat.liststrgvarb = gdat.listvalu.keys()
    
    numbvarb = len(gdat.liststrgvarb) 
    indxvarb = np.arange(numbvarb)
    
    gdat.numbvalu = np.empty(numbvarb, dtype=int)
    gdat.indxvalu = [[] for o in indxvarb]
    for o, strgvarb in enumerate(gdat.liststrgvarb):
        gdat.numbvalu[o] = len(gdat.listvalu[strgvarb])
        gdat.indxvalu[o] = np.arange(gdat.numbvalu[o])
    
    # dictionary to hold the metrics resulting from the runs
    dictmetr = {}
    liststrgmetr = ['prec', 'accu', 'reca']
    listlablmetr = ['Precision', 'Accuracy', 'Recall']
    liststrgrtyp = ['vali', 'tran']
    listlablrtyp = ['Training', 'Validation']
    numbrtyp = len(liststrgrtyp)
    gdat.indxrtyp = np.arange(numbrtyp)
    
    for o, strgvarb in enumerate(gdat.liststrgvarb):
        dictmetr[strgvarb] = np.empty((2, 3, numbruns, gdat.numbvalu[o]))
    
    # for each run
    for t in indxruns:
        
        print ('Run index %d' % t)
        # do the training for the central value
        #metr = functopo(gdat)
        
        # for each variable
        for o, strgvarb in enumerate(gdat.liststrgvarb): 
            
            print ('Processing variable %s...' % strgvarb)

            # for each value
            for i in gdat.indxvalu[o]:
              
                # temp -- this runs the central value redundantly and can be sped up by only running the central value once for all variables
                # do the training for the specific value of the variable of interest
                metr = functopo(gdat, i, strgvarb)
                
                dictmetr[strgvarb][0, 0, t, i] = metr[-1, 0, 0]
                dictmetr[strgvarb][1, 0, t, i] = metr[-1, 1, 0]
                dictmetr[strgvarb][0, 1, t, i] = metr[-1, 0, 1]
                dictmetr[strgvarb][1, 1, t, i] = metr[-1, 1, 1]
                dictmetr[strgvarb][0, 2, t, i] = metr[-1, 0, 2]
                dictmetr[strgvarb][1, 2, t, i] = metr[-1, 1, 2]
    
    alph = 0.5
    # plot the resulting metrics
    for o, strgvarb in enumerate(gdat.liststrgvarb): 
        for l, strgmetr in enumerate(liststrgmetr):
            figr, axis = plt.subplots() # figr unused
            
            for r in gdat.indxrtyp:
                yerr = np.empty((2, gdat.numbvalu[o]))
                if r == 0:
                    colr = 'b'
                else:
                    colr = 'g'
                
                indx = []
                ydat = np.empty(gdat.numbvalu[o])
                for i in gdat.indxvalu[o]:
                    indx.append(np.where(dictmetr[strgvarb][r, l, :, i] != -1)[0])
                    ydat[i] = np.mean(dictmetr[strgvarb][r, l, :, indx[i]], axis=0)
                if indx.size > 0:
                    for i in gdat.indxvalu[o]:
                        yerr[0, i] = ydat[i] - np.percentile(dictmetr[strgvarb][r, l, indx[i], i], 5.)
                        yerr[1, i] = np.percentile(dictmetr[strgvarb][r, l, :, i], 95.) - ydat[i]
                
                    if r == 0:
                        linestyl = '--' # unused
                    else:
                        linestyl = ''
                
                temp, listcaps, temp = axis.errorbar(gdat.listvalu[strgvarb], ydat, yerr=yerr, label=listlablrtyp[r], capsize=10, marker='o', \
                                                                                    ls='', markersize=10, lw=3, alpha=alph, color=colr)
                for caps in listcaps:
                    caps.set_markeredgewidth(3)
            
                for t in indxruns:
                    axis.plot(gdat.listvalu[strgvarb], dictmetr[strgvarb][r, l, t, :], marker='D', ls='', markersize=5, alpha=alph, color=colr)
            
            axis.set_ylim([0., 1.])

            if strgvarb == 'numbtime':
                labl = '$N_{time}$'
            
            if strgvarb == 'dept':
                labl = r'$\delta$' # pylint told me that these needed an r prefix
            
            if strgvarb == 'nois':
                labl = r'$\sigma$' # pylint told me that these needed an r prefix
            
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
            
            if strgvarb in ['numbdata', 'numbtime', 'dept', 'nois', 'numbdimsfrst', 'numbdimsseco', 'numbdatabtch']:
                axis.set_xscale('log')
            plt.legend()
            plt.tight_layout()
            path = pathplot + strgvarb + strgmetr + '_' + strgtimestmp + '.pdf' 
            plt.savefig(path)
            plt.close()
    

def cnfg_tansu():
    
    expl( \
         datatype='ete6', \
        )

