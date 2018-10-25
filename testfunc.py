import numpy as np
from sklearn.metrics import confusion_matrix
import datetime, os
import sys
from PIL import Image
import astropy as ap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from exop import main as exopmain
from garbag import models


def retrmetr(dataclass, indxvaluthis=None, strgvarbthis=None):
    xx = dataclass.numbepoc
    metr = np.zeros((xx, 2, 3)) -1 # size is numbepoc, 2, 3
    loss = np.empty(dataclass.numbepoc) # size is numbepoc

    numbepocchec = 5 # ??? this is hard-coded!

    for y in dataclass.indxepoc:
        # print(dataclass.modl.summary())
        # print('dataclass.inpt\n', dataclass.inpt.shape)
        
        histinpt = dataclass.inpt[:, :]    # instead of updating dataclass.inpt, which makes the size increase per call
        # print('histinpt\n', histinpt.shape)
        # print('outp ', dataclass.outp.shape)

        hist = dataclass.modl.fit(dataclass.inpt, dataclass.outp, epochs=dataclass.numbepoc, batch_size=dataclass.numbdatabtch, validation_split=dataclass.fractest, verbose=1)
        
        """
        loss[y] = hist.history['loss'][0]
        indxepocloww = max(0, y - numbepocchec)
        if y == dataclass.numbepoc - 1 and 100. * (loss[indxepocloww] - loss[y]):
            print('Warning! The optimizer may not have converged.')
            print('loss[indxepocloww]\n', loss[indxepocloww], '\nloss[y]\n', loss[y], '\nloss\n', loss)

        for r in dataclass.indxrtyp:
            if r==0:
                inpt = dataclass.inpttran
                outp = dataclass.outptran
                numdatatemp = dataclass.numbdatatran
            else:
                inpt = dataclass.inpttest
                outp = dataclass.outptest
                numbdatatemp = dataclass.numbdatatest

            inpt = inpt[:, :]
            outppred = (dataclass.modl.predict(inpt) > 0.5).astype(int)
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
                print ('No positive found...')
                #raise Exception('')
            metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne)
            if float(trpo + flne) > 0:
                metr[y, r, 2] = trpo / float(trpo + flne)
            else:
                raise Exception('')"""
            
    return metr

def explore(dataclass, modelfunc, datatype='here'):

    '''
    Function to explore the effect of hyper-parameters (and data properties for mock data) on binary classification metrics
    '''
    
    # global object that will hold global variables
    # this can be wrapped in a function to allow for customization 
    # initialize the data here
    gdat = dataclass

    ## time stamp string
    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # print ('CtC explorer initialized at %s.' % strgtimestmp)
    
    ## path where plots will be generated
    pathplot = os.environ['TDGU_DATA_PATH'] + '/'
    
    # print ('Will generate plots in %s' % pathplot)
    
    """"
    # detect names of devices, disabled for the moment
    from tensorflow.python.client import device_lib
    listdictdevi = device_lib.list_local_devices()
    print ('Names of the devices detected: ')
    for dictdevi in listdictdevi:
        print (dictdevi.name)
    """
    
    # temp
    gdat.maxmindxvarb = 10

    # for each run
    for t in gdat.indxruns:
        
        # print ('Run index %d' % t)
        # do the training for the central value
        # temp -- current implementation repeats running of the central point
        #metr = gdat.retr_metr()
        
        # for each variable
        for o, strgvarb in enumerate(gdat.liststrgvarb): 
            
            if o == gdat.maxmindxvarb:
                break

            # print ('Processing variable %s...' % strgvarb)

            # for each value
            for i in gdat.indxvalu[o]:
                
                pathsave = pathplot + '%04d%04d%04d.fits' % (t, o, i)
                # temp
                if False and os.path.exists(pathsave):
                    # print ('Reading %s...' % pathsave)
                    listhdun = ap.io.fits.open(pathsave)
                    metr = listhdun[0].data
                else:
                    for strgvarbtemp in gdat.liststrgvarb: 
                        setattr(gdat, strgvarbtemp, gdat.listvalu[strgvarbtemp][int(gdat.numbvalu[o]/2)])
                    setattr(gdat, strgvarb, gdat.listvalu[strgvarb][i])
                    
                    for strgvarbtemp in gdat.liststrgvarb: 
                        print ('strgvarbtemp, ', strgvarbtemp, ' gdat.strgvarbtemp, ', getattr(gdat, strgvarbtemp))

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
                    
                    # print ('Beginning')
                    # print ('gdat.inpt\n', gdat.inpt.shape)
                    
                    """
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
                    plt.xlabel('time')
                    plt.ylabel('data-input')
                    plt.title('input vs time')
                    plt.legend()
                    path = pathplot + 'inpt_%04d%s%04d' % (t, strgvarb, i) + strgtimestmp + '.pdf' 
                    plt.savefig(path)
                    plt.close()
                    """

                    # divide the data set into training and test data sets
                    numbdatatest = int(gdat.fractest * gdat.numbdata)
                    gdat.inpttest = gdat.inpt[:numbdatatest, :]
                    gdat.outptest = gdat.outp[:numbdatatest]
                    gdat.inpttran = gdat.inpt[numbdatatest:, :]
                    gdat.outptran = gdat.outp[numbdatatest:]   
                    
                    gdat.modl = modelfunc(gdat, )

                    # temp -- this runs the central value redundantly and can be sped up by only running the central value once for all variables
                    # do the training for the specific value of the variable of interest
                    metr = retrmetr(gdat, i, strgvarb)

                    """
                    # save to the disk
                    hdun = ap.io.fits.PrimaryHDU(metr)
                    listhdun = ap.io.fits.HDUList([hdun])
                    listhdun.writeto(pathsave, overwrite=True)
                    """

                gdat.dictmetr[strgvarb][0, 0, t, i] = metr[-1, 0, 0]
                gdat.dictmetr[strgvarb][1, 0, t, i] = metr[-1, 1, 0]
                gdat.dictmetr[strgvarb][0, 1, t, i] = metr[-1, 0, 1]
                gdat.dictmetr[strgvarb][1, 1, t, i] = metr[-1, 1, 1]
                gdat.dictmetr[strgvarb][0, 2, t, i] = metr[-1, 0, 2]
                gdat.dictmetr[strgvarb][1, 2, t, i] = metr[-1, 1, 2]

    return strgtimestmp

def plotter(dataclass, strgtimestmp=None):
    
    gdat = dataclass # for simplicity (moved the code)
    
    if strgtimestmp == None:
        strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    
    pathplot = os.environ['TDGU_DATA_PATH'] + '/'


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
                labl = '$\delta$'
            
            if strgvarb == 'nois':
                labl = '$\sigma$'
            
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
    

def metrics_vary_thresh(dataclass, points=50, indxvaluthis=None, strgvarbthis=None):
    num_thresholds = points
    pointlist = []
    thresholds = [0.3 + i/100 for i in range(points)]

    for y in dataclass.indxepoc:
        modelinst = dataclass.modl
        modelinst.fit(dataclass.inpt, dataclass.outp, epochs=dataclass.numbepoc, batch_size=dataclass.numbdatabtch, validation_split=dataclass.fractest, verbose=1)

        for r in dataclass.indxrtyp:
            if r==0:
                inpt = dataclass.inpttran
                outp = dataclass.outptran
            else:
                inpt = dataclass.inpttest
                outp = dataclass.outptest

            inpt = inpt[:, :]

             #for i in range(num_thresholds):
            for i in thresholds:
                # thresh = (i+1)/num_thresholds

                outppred = (modelinst.predict(inpt) > i).astype(int)
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
                    Precision = trpo / float(trpo + flpo) # precision
                else:
                    Precision = 0
                    # print ('No positive found...')
                    # raise Exception('')
                # metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne) # accuracy
                if float(trpo + flne) > 0:
                    Recall = trpo / float(trpo + flne) # recall
                else:
                    Recall = 0
                    # raise Exception('')
                
                pointlist.append((Precision, Recall))
    return pointlist

def Precision_Recall_Ehhh(dataclass, strgtimestmp=None):
    
    gdat = dataclass # for simplicity (moved the code)
    
    if strgtimestmp == None:
        strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    
    pathplot = os.environ['TDGU_DATA_PATH'] + '/'


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
                labl = '$\delta$'
            
            if strgvarb == 'nois':
                labl = '$\sigma$'
            
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
 
def explore_and_save(dataclass, modelfunc, datatype='here'):

    '''
    Function to explore the effect of hyper-parameters (and data properties for mock data) on binary classification metrics
    '''
    
    # global object that will hold global variables
    # this can be wrapped in a function to allow for customization 
    # initialize the data here
    gdat = dataclass

    ## time stamp string
    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # print ('CtC explorer initialized at %s.' % strgtimestmp)
    
    ## path where plots will be generated
    pathplot = os.environ['TDGU_DATA_PATH'] + '/'
    
    # print ('Will generate plots in %s' % pathplot)
    
    """"
    # detect names of devices, disabled for the moment
    from tensorflow.python.client import device_lib
    listdictdevi = device_lib.list_local_devices()
    print ('Names of the devices detected: ')
    for dictdevi in listdictdevi:
        print (dictdevi.name)
    """
    
    # temp
    gdat.maxmindxvarb = 10

    # for each run
    for t in gdat.indxruns:
        
        # print ('Run index %d' % t)
        # do the training for the central value
        # temp -- current implementation repeats running of the central point
        #metr = gdat.retr_metr()
        
        # for each variable
        for o, strgvarb in enumerate(gdat.liststrgvarb): 
            
            if o == gdat.maxmindxvarb:
                break

            # print ('Processing variable %s...' % strgvarb)

            # for each value
            for i in gdat.indxvalu[o]:
                
                pathsave = pathplot + '%04d%04d%04d.fits' % (t, o, i)
                # temp
                if False and os.path.exists(pathsave):
                    # print ('Reading %s...' % pathsave)
                    listhdun = ap.io.fits.open(pathsave)
                    metr = listhdun[0].data
                else:
                    for strgvarbtemp in gdat.liststrgvarb: 
                        setattr(gdat, strgvarbtemp, gdat.listvalu[strgvarbtemp][int(gdat.numbvalu[o]/2)])
                    setattr(gdat, strgvarb, gdat.listvalu[strgvarb][i])
                    
                    # for strgvarbtemp in gdat.liststrgvarb: 
                    #print (strgvarb, getattr(gdat, strgvarb))

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
                    
                    # print ('Beginning')
                    # print ('gdat.inpt\n', gdat.inpt.shape)
                    
                    """
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
                    plt.xlabel('time')
                    plt.ylabel('data-input')
                    plt.title('input vs time')
                    plt.legend()
                    path = pathplot + 'inpt_%04d%s%04d' % (t, strgvarb, i) + strgtimestmp + '.pdf' 
                    plt.savefig(path)
                    plt.close()
                    """

                    # divide the data set into training and test data sets
                    numbdatatest = int(gdat.fractest * gdat.numbdata)
                    gdat.inpttest = gdat.inpt[:numbdatatest, :]
                    gdat.outptest = gdat.outp[:numbdatatest]
                    gdat.inpttran = gdat.inpt[numbdatatest:, :]
                    gdat.outptran = gdat.outp[numbdatatest:]   
                    
                    gdat.modl = modelfunc(gdat, )

                    # temp -- this runs the central value redundantly and can be sped up by only running the central value once for all variables
                    # do the training for the specific value of the variable of interest
                    metr = retrmetr(gdat, i, strgvarb)
                    
                    gdat.modl.save('threeEpoch_{0}_{1}layer_{2}{3}.h5'.format(str('singleinput'), str(gdat.numblayr), str(strgvarb), str(getattr(gdat, strgvarb))))
                    """
                    # save to the disk
                    hdun = ap.io.fits.PrimaryHDU(metr)
                    listhdun = ap.io.fits.HDUList([hdun])
                    listhdun.writeto(pathsave, overwrite=True)
                    """
                """
                gdat.dictmetr[strgvarb][0, 0, t, i] = metr[-1, 0, 0]
                gdat.dictmetr[strgvarb][1, 0, t, i] = metr[-1, 1, 0]
                gdat.dictmetr[strgvarb][0, 1, t, i] = metr[-1, 0, 1]
                gdat.dictmetr[strgvarb][1, 1, t, i] = metr[-1, 1, 1]
                gdat.dictmetr[strgvarb][0, 2, t, i] = metr[-1, 0, 2]
                gdat.dictmetr[strgvarb][1, 2, t, i] = metr[-1, 1, 2]"""

    return strgtimestmp




def Precision_Recall(dataclass, points=50):

    # num_thresholds = points
    # pointlist = []

    for y in dataclass.indxepoc:
        modelinst = dataclass.modl
        modelinst.fit(dataclass.inpt, dataclass.outp, epochs=dataclass.numbepoc, batch_size=dataclass.numbdatabtch, validation_split=dataclass.fractest, verbose=1)

        for r in dataclass.indxrtyp:
            if r==0:
                inpt = dataclass.inpttran
                outp = dataclass.outptran
            else:
                inpt = dataclass.inpttest
                outp = dataclass.outptest

            inpt = inpt[:, :]

            # for i in range(num_thresholds):
            # thresh = (i+1)/num_thresholds

            outppred = (modelinst.predict(inpt) > 0.5).astype(int)
            matrconf = confusion_matrix(outp, outppred)

            if matrconf.size == 1:
                matrconftemp = np.copy(matrconf)
                matrconf = np.empty((2, 2))
                matrconf[0, 0] = matrconftemp

            trne = matrconf[0, 0]
            flpo = matrconf[0, 1]
            flne = matrconf[1, 0]
            trpo = matrconf[1, 1]

            print(matrconf)

            if float(trpo + flpo) > 0:
                Precision = trpo / float(trpo + flpo) # precision
            else:
                Precision = 0
                # print ('No positive found...')
                #raise Exception('')
            # metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne) # accuracy
            if float(trpo + flne) > 0:
                Recall = trpo / float(trpo + flne) # recall
            else:
                Recall = 0
                # raise Exception('')
            
            # pointlist.append((Precision, Recall))
    return Precision, Recall


def run_through_puts(dataclass, modelfunc, datatype='here'):
    
    '''
    Function to explore the effect of hyper-parameters (and data properties for mock data) on binary classification metrics
    '''
    
    # global object that will hold global variables
    # this can be wrapped in a function to allow for customization 
    # initialize the data here
    gdat = dataclass

    ## time stamp string
    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # print ('CtC explorer initialized at %s.' % strgtimestmp)
    
    ## path where plots will be generated
    pathplot = os.environ['TDGU_DATA_PATH'] + '/'
    
    # temp
    gdat.maxmindxvarb = 10

    # for each run
    for t in gdat.indxruns:
        
        # for each variable
        for o, strgvarb in enumerate(gdat.liststrgvarb): 
            
            if o == gdat.maxmindxvarb:
                break

            pr_points = []

            # for each value
            for i in gdat.indxvalu[o]:

                for strgvarbtemp in gdat.liststrgvarb: 
                    setattr(gdat, strgvarbtemp, gdat.listvalu[strgvarbtemp][int(gdat.numbvalu[o]/2)])
                setattr(gdat, strgvarb, gdat.listvalu[strgvarb][i])
                
                # for strgvarbtemp in gdat.liststrgvarb: 
                #print (strgvarb, getattr(gdat, strgvarb))

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


                # divide the data set into training and test data sets
                numbdatatest = int(gdat.fractest * gdat.numbdata)
                gdat.inpttest = gdat.inpt[:numbdatatest, :]
                gdat.outptest = gdat.outp[:numbdatatest]
                gdat.inpttran = gdat.inpt[numbdatatest:, :]
                gdat.outptran = gdat.outp[numbdatatest:]   
                
                gdat.modl = modelfunc(gdat, )

                # precision, recall = Precision_Recall(gdat)

                # pr_points.append((precision, recall))

                pr_points = metrics_vary_thresh(gdat)
                
            figr, axis = plt.subplots()
            axis.plot([i[0] for i in pr_points], [i[1] for i in pr_points], marker='o', ls='', markersize=5, alpha=0.6)
            plt.tight_layout()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision v Recall, {0}{1}'.format(str(strgvarb), str(getattr(gdat, strgvarb))))
            # plt.legend()
            path = pathplot + 'PvR_{0}_{1}{2}_'.format(t, strgvarb, getattr(gdat, strgvarb)) + strgtimestmp + '.pdf' 
            plt.savefig(path)
            plt.close()
                


                    

    return strgtimestmp

