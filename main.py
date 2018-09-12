import numpy as np

import datetime, os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
import tensorflow as tf

import sklearn
from sklearn.metrics import confusion_matrix

import astropy as ap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)
import exop
from exop import main as exopmain

class gdatstrt(object):
    
    """
    init: Initializes all the testing data -- has all variables needed for testing
    appdfcon: add a fully connected layer
    appdcon1: add a 1D convolutional layer
    retr_metr: returns all metrics of the network
    """
    
    def __init__(self, datatype='here'):
    
        # fraction of data samples that will be used to test the model
        self.fractest = 0.1
    
        # number of epochs
        self.numbepoc = 10
    
        # number of runs for each configuration in order to determine the statistical uncertainty
        self.numbruns = 3

        self.indxepoc = np.arange(self.numbepoc)
        self.indxruns = np.arange(self.numbruns)

        # a dictionary to hold the variable values for which the training will be repeated
        self.listvalu = {}
        ## generative parameters of mock data
        self.listvalu['numbtime'] = np.array([1e1, 3e1, 1e2, 3e2, 1e3]).astype(int)
        # temp
        self.listvalu['dept'] = 1 - np.array([1e-3, 3e-3, 3e-1, 3e-2, 1e-1])
        self.listvalu['nois'] = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        self.listvalu['numbdata'] = np.array([3e1, 1e4, 3e4, 1e5, 3e5]).astype(int)
        self.listvalu['fracplan'] = [0.1, 0.3, 0.5, 0.7, 0.9]
        ## hyperparameters
        self.listvalu['numbdatabtch'] = [16, 32, 64, 128, 256]
        ### number of layers
        # temp
        self.listvalu['numblayr'] = [1, 2, 3, 4, 5]
        ### number of dimensions in each layer
        # temp
        self.listvalu['numbdimslayr'] = [32, 64, 128, 256, 512]
        ### fraction of dropout in in each layer
        # temp
        self.listvalu['fracdrop'] = [0.25, 0.4, 0.5, 0.6, 0.75]
        
        # list of strings holding the names of the variables
        self.liststrgvarb = self.listvalu.keys()
        
        self.numbvarb = len(self.liststrgvarb) # number of variables
        self.indxvarb = np.arange(self.numbvarb) # array of all indexes to get any variable
        
        self.numbvalu = np.empty(self.numbvarb, dtype=int)
        self.indxvalu = [[] for o in self.indxvarb]
        for o, strgvarb in enumerate(self.liststrgvarb):
            self.numbvalu[o] = len(self.listvalu[strgvarb])
            self.indxvalu[o] = np.arange(self.numbvalu[o])
        
        # dictionary to hold the metrics resulting from the runs
        self.dictmetr = {}
        self.liststrgmetr = ['prec', 'accu', 'reca']
        self.listlablmetr = ['Precision', 'Accuracy', 'Recall']
        self.liststrgrtyp = ['vali', 'tran']
        self.listlablrtyp = ['Training', 'Validation']
        self.numbrtyp = len(self.liststrgrtyp)
        self.indxrtyp = np.arange(self.numbrtyp)
        
        for o, strgvarb in enumerate(self.liststrgvarb):
            self.dictmetr[strgvarb] = np.empty((2, 3, self.numbruns, self.numbvalu[o]))

    
    # trying to condense all class things into one __init__ so all methods can just be called here
    def appdfcon(self, fracdrop, strglayr='init'):
        """
        Functionally can be added at any point in the model

        fracdrop: fraction of drop-out
        strglayr: 'init', 'medi', 'finl'
        """

        if strglayr == 'init':
            self.modl.add(Dense(self.numbdimslayr, input_dim=self.numbtime, activation='relu'))
        elif strglayr == 'inte':
            self.modl.add(Dense(self.numbdimslayr, activation= 'relu'))
        elif strglayr == 'last':
            self.modl.add(Dense(1, activation='sigmoid'))
        
        if fracdrop > 0.:
            self.modl.add(Dropout(self.fracdrop))
        

    def appdcon1(self, fracdrop, strglayr='init', strgactv='relu'):
        """
        Adds a 1D CNN layer to the network
        This should not be the last layer!

        fracdrop: fraction of drop-out
        strglayr: 'init', 'medi', 'finl'
        """
        
        if strglayr == 'init':
            self.modl.add(Conv1D(self.numbdimslayr, kernel_size=self.numbtime, input_dim=self.numbtime, activation='relu'))
        elif strglayr == 'medi':
            self.modl.add(Conv1D(self.numbdimslayr, kernel_size=self.numbtime, activation= 'relu'))
            
        if fracdrop > 0.:
            self.modl.add(Dropout(self.fracdrop))
        


    def retr_metr(self, indxvaluthis=None, strgvarbthis=None):     
        """
        Performance method
        """

        # empt dict
        listvalutemp = {}
        # store with the vars we iterate over
        for o, strgvarb in enumerate(self.liststrgvarb):
            listvalutemp[strgvarb] = self.listvalu[strgvarb][self.numbvalu[o]/2]        
        
        # catch that input and set another val in the dict
        if strgvarbthis != None:
            listvalutemp[strgvarbthis] = self.listvalu[strgvarbthis][indxvaluthis]   

        metr = np.zeros((self.numbepoc, 2, 3)) - 1
        loss = np.empty(self.numbepoc)
        numbepocchec = 5 # hard coded
        
        for y in self.indxepoc:
            hist = self.modl.fit(self.inpt, self.outp, epochs=1, batch_size=self.numbdatabtch, validation_split=self.fractest, verbose=1)
            loss[y] = hist.history['loss'][0]
            indxepocloww = max(0, y - numbepocchec)
            if y == self.numbepoc - 1 and 100. * (loss[indxepocloww] - loss[y]):
                print('Warning! The optimizer may not have converged.')
                print('loss[indxepocloww]\n', loss[indxepocloww], '\nloss[y]\n', loss[y], '\nloss\n', loss)

            for r in self.indxrtyp:
                if r==0:
                    inpt = self.inpttran
                    outp = self.outptran
                    numdatatemp = self.numbdatatran
                else:
                    inpt = self.inpttest
                    outp = self.outptest
                    numbdatatemp = self.numbdatatest

                outppred = (self.modl.predict(inpt) > 0.5).astype(int)
                matrconf = confusion_matrix(outp, outppred)
                if matrconf.size == 1:
                    matrconftemp = np.copy(matrconf)
                    matrconf = np.empty((2, 2))
                    matrconf[0, 0] = matrconftemp
                trne = matrconf[0, 0]
                flpo = matrconf[0, 1]
                flne = matrconf[1, 0]
                trpo = matrconf[1, 1]

                if float(trpo + flpo) > 0:
                    metr[y, r, 0] = trpo / float(trpo + flpo)
                else:
                    print 'No positive found...'
                    #raise Exception('')
                metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne)
                if float(trpo + flne) > 0:
                    metr[y, r, 2] = trpo / float(trpo + flne)
                else:
                    raise Exception('')
                
        return metr


def summgene(varb):
    '''
    convenience function to quickly print a numpy array
    '''
    
    print (np.amin(varb))
    print (np.amax(varb))
    print (np.mean(varb))
    print (varb.shape)


def expl( \
         # string indicating the model
         strguser='tansu', \
         strgtopo='fcon', \
         # if local, operates normal, if local+globa or dub(double) it will take local and global at the same time
         zoomtype='locl', \
         datatype='here', \
        ):

    '''
    Function to explore the effect of hyper-parameters (and data properties for mock data) on binary classification metrics
    '''
    
    # initialization
    ## global object that will hold global variables
    gdat = gdatstrt()

    ## time stamp string
    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print ('CtC explorer initialized at %s.' % strgtimestmp)
    
    ## path where plots will be generated
    pathplot = os.environ['TDGU_DATA_PATH'] + '/nnet_ssupgadn/'
    
    print ('Will generate plots in %s' % pathplot)
    
    """"
    # detect names of devices, disabled for the moment
    from tensorflow.python.client import device_lib
    listdictdevi = device_lib.list_local_devices()
    print ('Names of the devices detected: ')
    for dictdevi in listdictdevi:
        print (dictdevi.name)
    """
    
    # temp
    gdat.maxmindxvarb = 10000

    # for each run
    for t in gdat.indxruns:
        
        print ('Run index %d' % t)
        # do the training for the central value
        # temp -- current implementation repeats running of the central point
        #metr = gdat.retr_metr()
        
        # for each variable
        for o, strgvarb in enumerate(gdat.liststrgvarb): 
            
            if o == gdat.maxmindxvarb:
                break

            print ('Processing variable %s...' % strgvarb)

            # for each value
            for i in gdat.indxvalu[o]:
                
                pathsave = pathplot + '%04d%04d%04d.fits' % (t, o, i)
                # temp
                if False and os.path.exists(pathsave):
                    listhdun = ap.io.fits.open(pathsave)
                    metr = listhdun[0].data 
                else:
                    for strgvarbtemp in gdat.liststrgvarb: 
                        setattr(gdat, strgvarbtemp, gdat.listvalu[strgvarbtemp][gdat.numbvalu[o]/2])
                    setattr(gdat, strgvarb, gdat.listvalu[strgvarb][i])
                    
                    for strgvarbtemp in gdat.liststrgvarb: 
                        print (strgvarbtemp)
                        print (getattr(gdat, strgvarbtemp))

                    gdat.numbplan = int(gdat.numbdata * gdat.fracplan)
                    gdat.numbnois = gdat.numbdata - gdat.numbplan
                    
                    gdat.indxtime = np.arange(gdat.numbtime)
                    gdat.indxdata = np.arange(gdat.numbdata)
                    gdat.indxlayr = np.arange(gdat.numblayr)

                    # number of test data samples
                    gdat.numbdatatest = int(gdat.numbdata * gdat.fractest)
                    # number of training data samples
                    gdat.numbdatatran = gdat.numbdata - gdat.numbdatatest
                    # number of signal data samples
                    numbdataplan = int(gdat.numbdata * gdat.fracplan)
                    
                    if datatype == 'here':
                        gdat.inpt, gdat.outp = exopmain.retr_datamock(numbplan=gdat.numbplan, numbnois=gdat.numbnois, numbtime=gdat.numbtime, dept=gdat.dept, nois=gdat.nois)

                    if datatype == 'ete6':
                        gdat.inpt, gdat.outp = exopmain.retr_ete6()
                        
                    # plot
                    figr, axis = plt.subplots() # figr unused
                    for k in gdat.indxdata:
                        if k < 10:
                            if gdat.outp[k] == 1:
                                colr = 'r'
                            else:
                                colr = 'b'
                            axis.plot(gdat.indxtime, gdat.inpt[k, :], marker='o', ls='-', markersize=5, alpha=0.6, color=colr)
                    plt.tight_layout()
                    path = pathplot + 'inpt_%04d%s%04d' % (t, strgvarb, i) + strgtimestmp + '.pdf' 
                    plt.savefig(path)
                    plt.close()
        
                    # divide the data set into training and test data sets
                    numbdatatest = int(gdat.fractest * gdat.numbdata)
                    gdat.inpttest = gdat.inpt[:numbdatatest, :]
                    gdat.outptest = gdat.outp[:numbdatatest]
                    gdat.inpttran = gdat.inpt[numbdatatest:, :]
                    gdat.outptran = gdat.outp[numbdatatest:]   

                    gdat.modl = Sequential()

                    # construct the neural net
                    # add the first fully connected layer
                    gdat.appdfcon(gdat.fracdrop)
                    
                    ## add other fully connected layers
                    if gdat.numblayr > 2:
                        for k in range(gdat.numblayr - 2):
                            gdat.appdfcon(gdat.fracdrop, strglayr='inte')
                    
                    ## add the last output layer
                    gdat.appdfcon(gdat.fracdrop, strglayr='last')
                    
                    gdat.modl.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
                    
                    # temp -- this runs the central value redundantly and can be sped up by only running the central value once for all variables
                    # do the training for the specific value of the variable of interest
                    metr = gdat.retr_metr(i, strgvarb)
                    
                    # save to the disk
                    hdun = ap.io.fits.PrimaryHDU(metr)
                    listhdun = ap.io.fits.HDUList([hdun])
                    listhdun.writeto(pathsave, overwrite=True)

                gdat.dictmetr[strgvarb][0, 0, t, i] = metr[-1, 0, 0]
                gdat.dictmetr[strgvarb][1, 0, t, i] = metr[-1, 1, 0]
                gdat.dictmetr[strgvarb][0, 1, t, i] = metr[-1, 0, 1]
                gdat.dictmetr[strgvarb][1, 1, t, i] = metr[-1, 1, 1]
                gdat.dictmetr[strgvarb][0, 2, t, i] = metr[-1, 0, 2]
                gdat.dictmetr[strgvarb][1, 2, t, i] = metr[-1, 1, 2]
    
    alph = 0.5
    # plot the resulting metrics
    for o, strgvarb in enumerate(gdat.liststrgvarb): 
        
        if o == gdat.maxmindxvarb:
            break

        for l, strgmetr in enumerate(gdat.liststrgmetr):
            figr, axis = plt.subplots() # figr unused
            
            for r in gdat.indxrtyp:
                yerr = np.zeros((2, gdat.numbvalu[o]))
                if r == 0:
                    colr = 'b'
                else:
                    colr = 'g'
                
                indx = []
                ydat = np.zeros(gdat.numbvalu[o]) - 1.
                for i in gdat.indxvalu[o]:
                    indx.append(np.where(gdat.dictmetr[strgvarb][r, l, :, i] != -1)[0])
                    if indx[i].size > 0:
                        ydat[i] = np.mean(gdat.dictmetr[strgvarb][r, l, indx[i], i], axis=0)
                        yerr[0, i] = ydat[i] - np.percentile(gdat.dictmetr[strgvarb][r, l, indx[i], i], 5.)
                        yerr[1, i] = np.percentile(gdat.dictmetr[strgvarb][r, l, indx[i], i], 95.) - ydat[i]
                
                temp, listcaps, temp = axis.errorbar(gdat.listvalu[strgvarb], ydat, yerr=yerr, label=gdat.listlablrtyp[r], capsize=10, marker='o', \
                                                                                    ls='', markersize=10, lw=3, alpha=alph, color=colr)
                
                for caps in listcaps:
                    caps.set_markeredgewidth(3)
            
                for t in gdat.indxruns:
                    axis.plot(gdat.listvalu[strgvarb], gdat.dictmetr[strgvarb][r, l, t, :], marker='D', ls='', markersize=5, alpha=alph, color=colr)
            
            #axis.set_ylim([-0.1, 1.1])

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
        
            if strgvarb == 'numbdimslayr':
                labl = '$N_{dens}$'
        
            if strgvarb == 'fracdrop':
                labl = '$f_D$'
        
            axis.set_ylabel(gdat.listlablmetr[l]) 
            axis.set_xlabel(labl) 
            
            if strgvarb in ['numbdata', 'numbtime', 'dept', 'nois', 'numbdimslayr', 'numbdatabtch']:
                axis.set_xscale('log')
            plt.legend()
            plt.tight_layout()
            path = pathplot + strgvarb + strgmetr + '_' + strgtimestmp + '.pdf' 
            plt.savefig(path)
            plt.close()
    

