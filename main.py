import numpy as np

import datetime, os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import keras.utils
from keras import backend as K

import tensorflow as tf

import lightkurve

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
    
    def __init__(self):
        pass
    

def appdfcon(gdat):
    
    if gdat.numblayr == 1:
        gdat.modl.add(Dropout(gdat.fracdrop))
        gdat.modl.add(Dense(1, input_dim=gdat.numbphas, activation='sigmoid'))
    else:
        gdat.modl.add(Dropout(gdat.fracdrop))
        gdat.modl.add(Dense(gdat.numbdimslayr, input_dim=gdat.numbphas, activation='relu'))
        for k in range(gdat.numblayr):
            gdat.modl.add(Dropout(gdat.fracdrop))
            gdat.modl.add(Dense(gdat.numbdimslayr, activation= 'relu'))
        gdat.modl.add(Dropout(gdat.fracdrop))
        gdat.modl.add(Dense(1, activation='sigmoid'))
    

def appdcon1(gdat, strgactv='relu'):
    
    gdat.modl.add(Conv1D(16, kernel_size=5, input_shape=(gdat.numbphas, 1), activation='relu', padding='same'))
    gdat.modl.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    gdat.modl.add(Conv1D(16, kernel_size=5, activation='relu', padding='same'))
    gdat.modl.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    gdat.modl.add(Conv1D(16, kernel_size=5, activation='relu', padding='same'))
    gdat.modl.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    gdat.modl.add(Conv1D(16, kernel_size=5, activation='relu', padding='same'))
    gdat.modl.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    gdat.modl.add(Flatten())


def retr_metr(gdat, indxvaluthis=None, strgvarbthis=None):     
    
    """
    Calculates the binary classification metrics such as accuracy, recall and precision
    """

    metr = np.zeros((gdat.numbepoc, 2, 3 )) - 1

    loss = np.empty(gdat.numbepoc)
    numbepocchec = 5
    
    print gdat.modl.summary()
    for y in gdat.indxepoc:
        print 'Training epoch %d...' % y
        histinpt = gdat.inpttran[:, :, None]
        hist = gdat.modl.fit(histinpt, gdat.outptran, epochs=1, batch_size=gdat.numbdatabtch, verbose=1)
        loss[y] = hist.history['loss'][0]
        indxepocloww = max(0, y - numbepocchec)
        
        for layr in gdat.modl.layers:
            func = keras.backend.function([gdat.modl.input, keras.backend.learning_phase()], [layr.output])
            
            listweigbias = layr.get_weights()
            #assert len(listweigbias) == 2
            print 'listweigbias'
            for n in range(len(listweigbias)):
                print 'n'
                print n
                print 'listweigbias[n]'
                summgene(listweigbias[n])
            stat = func([histinpt, 1.])
            print 'type(stat)'
            print type(stat)
            print 'len(stat)'
            print len(stat)
            for n in range(len(stat)):
                print 'stat[n]'
                summgene(stat[n])
                print
            print


        if y == gdat.numbepoc - 1 and 100. * (loss[indxepocloww] - loss[y]):
            print 'Warning! The optimizer may not have converged.'
            print 'loss[indxepocloww]\n', loss[indxepocloww], '\nloss[y]\n', loss[y], '\nloss\n', loss

        for r in gdat.indxrtyp:
            if r == 0:
                inpt = gdat.inpttran
                outp = gdat.outptran
                numdatatemp = gdat.numbdatatran
            else:
                inpt = gdat.inpttest
                outp = gdat.outptest
                numbdatatemp = gdat.numbdatatest
            inpt = inpt[:, :, None]
            
            outppredsigm = gdat.modl.predict(inpt)
            outppred = (outppredsigm > 0.5).astype(int)
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
                metr[y, r, 0] = trpo / float(trpo + flpo) # precision
            else:
                pass
                #print ('No positive found...')
                #raise Exception('')
            metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne) # accuracy
            if float(trpo + flne) > 0:
                metr[y, r, 2] = trpo / float(trpo + flne) # recall
            else:
                print 'No relevant sample!'
                #raise Exception('')
            
            print 'metr[y, r, :]'
            print metr[y, r, :]
            print 
    return metr


def expl( \
         # string indicating the model
         strguser='tday', \
         strgtopo='fcon', \
         # if local, operates normal, if local+globa or dub(double) it will take local and global at the same time
         zoomtype='locl', \
         phastype='flbn', \
         datatype='simpmock', \
         #datatype='tess', \
):

    '''
    Function to explore the effect of hyper-parameters (and data properties for mock data) on binary classification metrics
    '''
    
    # global object that will hold global variables
    gdat = gdatstrt()
    
    gdat.datatype = datatype
    
    # Boolean flag to use light curves folded and binned  by SPOC
    if datatype == 'tess':
        gdat.boolspocflbn = True
    else:
        gdat.boolspocflbn = False
    
    # fraction of data samples that will be used to test the model
    gdat.fractest = 0.1
    
    # number of epochs
    gdat.numbepoc = 20
    
    # number of runs for each configuration in order to determine the statistical uncertainty
    gdat.numbruns = 1

    gdat.indxepoc = np.arange(gdat.numbepoc)
    gdat.indxruns = np.arange(gdat.numbruns)

    # a dictionary to hold the variable values for which the training will be repeated
    gdat.listvalu = {}
    # temp
    gdat.listvalu['dept'] = 1 - np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1]) 

    gdat.listvalu['zoomtype'] = ['locl', 'glob']
    
    gdat.numbtime = 10000

    if gdat.datatype == 'simpmock':

        ## generative parameters of mock data
        #gdat.listvalu['numbphas'] = np.array([1e1, 3e1, 1000, 3e2, 1e3]).astype(int)
        gdat.listvalu['numbphas'] = np.array([2000]).astype(int)
        # temp
        #gdat.listvalu['dept'] = np.array([1e-3, 3e-3, 3e-1, 3e-2, 1e-1]) 
        gdat.listvalu['dept'] = np.array([3e-1]) 
        #gdat.listvalu['nois'] = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1]) # SNR 
        gdat.listvalu['nois'] = np.array([1e-3, 1e-1, 1e1]) # SNR 
        #gdat.listvalu['numbrele'] = np.array([3e3, 1e4 , 10, 1e5, 3e5]).astype(int)
        gdat.listvalu['numbrele'] = np.array([300]).astype(int)
        #gdat.listvalu['numbirre'] = np.array([3e3, 1e4 , 100, 1e5, 3e5]).astype(int)
        gdat.listvalu['numbirre'] = np.array([300]).astype(int)

    else:
        ## generative parameters of mock data
        gdat.listvalu['numbphas'] = np.array([1e1, 3e1, 20076, 3e2, 1e3]).astype(int)
        ## generative parameters of mock data

        gdat.listvalu['numbrele'] = np.array([100]).astype(int)
        gdat.listvalu['numbirre'] = np.array([100]).astype(int)

    ## hyperparameters
    ### data augmentation
    #gdat.listvalu['zoomtype'] = ['locl', 'glob']
    gdat.listvalu['zoomtype'] = ['glob']
    ### neural network
    #### batch size
    #gdat.listvalu['numbdatabtch'] = [16, 32, 64, 128, 256]
    gdat.listvalu['numbdatabtch'] = [64]
    #### number of FC layers
    #gdat.listvalu['numblayr'] = [1, 2, 3, 4, 5]
    gdat.listvalu['numblayr'] = [1]
    #### number of dimensions in each layer
    #gdat.listvalu['numbdimslayr'] = [32, 64, 128, 256, 512]
    gdat.listvalu['numbdimslayr'] = [128]
    #### fraction of dropout in in each layer
    #gdat.listvalu['fracdrop'] = [0., 0.15, 0.3, 0.45, 0.6]
    gdat.listvalu['fracdrop'] = [0.3]
    
    # list of strings holding the names of the variables
    gdat.liststrgvarb = gdat.listvalu.keys()
    
    gdat.numbvarb = len(gdat.liststrgvarb) # number of variables
    gdat.indxvarb = np.arange(gdat.numbvarb) # array of all indexes to get any variable
    
    gdat.numbvalu = np.empty(gdat.numbvarb, dtype=int)
    gdat.indxvalu = [[] for o in gdat.indxvarb]
    for o, strgvarb in enumerate(gdat.liststrgvarb):
        gdat.numbvalu[o] = len(gdat.listvalu[strgvarb])
        gdat.indxvalu[o] = np.arange(gdat.numbvalu[o])
    
    # dictionary to hold the metrics resulting from the runs
    gdat.dictmetr = {}
    gdat.liststrgmetr = ['prec', 'accu', 'reca']
    gdat.listlablmetr = ['Precision', 'Accuracy', 'Recall']
    gdat.liststrgrtyp = ['vali', 'tran']
    gdat.listlablrtyp = ['Training', 'Validation']
    gdat.numbrtyp = len(gdat.liststrgrtyp)
    gdat.indxrtyp = np.arange(gdat.numbrtyp)
    
    for o, strgvarb in enumerate(gdat.liststrgvarb):
        gdat.dictmetr[strgvarb] = np.empty((2, 3, gdat.numbruns, gdat.numbvalu[o]))

    gdat.phastype = phastype

    ## time stamp string
    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print ('CtC explorer initialized at %s.' % strgtimestmp)
    
    ## path where plots will be generated
    pathplot = os.environ['CTHC_DATA_PATH'] + '/inpt/'
    os.system('mkdir -p %s' % pathplot)
    print ('Will generate plots in %s' % pathplot)
    
    # detect names of devices, disabled for the moment
    from tensorflow.python.client import device_lib
    listdictdevi = device_lib.list_local_devices()
    print ('Names of the devices detected: ')
    for dictdevi in listdictdevi:
        print (dictdevi.name)
    
    #gdat.numbphas = 20076
    #gdat.indxphas = np.arange(gdat.numbphas)

    # temp
    gdat.maxmindxvarb = 10

    # for each run
    for t in gdat.indxruns:
        
        print 'Run index %d...' % t
        # do the training for the central value
        # temp -- current implementation repeats running of the central point
        #metr = gdat.retr_metr(gdat)
        
        # for each variable
        for o, strgvarb in enumerate(gdat.liststrgvarb): 
            
            if o == gdat.maxmindxvarb:
                break
            
            if len(gdat.indxvalu[o]) == 1:
                continue

            print 'Processing variable %s...' % strgvarb

            # for each value
            for i in gdat.indxvalu[o]:
                
                strgconf = '%04d_%04d_%04d' % (t, o, i)
                pathsave = pathplot + 'save_metr_%s.fits' % strgconf
                # temp
                if False and os.path.exists(pathsave):
                    print ('Reading from %s...' % pathsave)
                    listhdun = ap.io.fits.open(pathsave)
                    metr = listhdun[0].data
                else:
                    for strgvarbtemp in gdat.liststrgvarb:
                        indx = int(len(gdat.listvalu[strgvarbtemp]) / 2)
                        setattr(gdat, strgvarbtemp, gdat.listvalu[strgvarbtemp][indx])
                    setattr(gdat, strgvarb, gdat.listvalu[strgvarb][i])
                    
                    if isinstance(gdat.listvalu[strgvarb][i], str):
                        print 'Value: ' + gdat.listvalu[strgvarb][i]
                    else:
                        print 'Value: %g' % gdat.listvalu[strgvarb][i]
                    
                    for strgvarbtemp in gdat.liststrgvarb: 
                        print (strgvarbtemp)
                        print (getattr(gdat, strgvarbtemp))

                    gdat.numbdata = gdat.numbrele + gdat.numbirre
                    gdat.fracrele = gdat.numbrele / float(gdat.numbdata)
                    
                    gdat.indxphas = np.arange(gdat.numbphas)
                    gdat.indxdata = np.arange(gdat.numbdata)
                    gdat.indxlayr = np.arange(gdat.numblayr)

                    # number of test data samples
                    gdat.numbdatatest = int(gdat.numbdata * gdat.fractest)
                    # number of training data samples
                    gdat.numbdatatran = gdat.numbdata - gdat.numbdatatest
                    
                    if datatype == 'simpmock':
                        gdat.inptraww, gdat.outp, gdat.peri = exopmain.retr_datamock(numbplan=gdat.numbrele, \
                                                    numbnois=gdat.numbirre, numbtime=gdat.numbtime, dept=gdat.dept, nois=gdat.nois)
                        gdat.time = np.tile(np.linspace(0., (gdat.numbtime - 1) / 30. / 24., gdat.numbtime), (gdat.numbdata, 1))
                        gdat.legdoutp = []
                        for k in gdat.indxdata:
                            legd = '%d, ' % k
                            if gdat.outp[k] == 1:
                                legd += 'R'
                            else:
                                legd += 'I'
                            gdat.legdoutp.append(legd)

                    if datatype == 'ete6':
                        gdat.time, gdat.inptraww, gdat.outp, gdat.tici, gdat.peri = exopmain.retr_dataete6(numbdata=gdat.numbdata, nois=gdat.nois)
                    
                    if datatype == 'tess':
                        if gdat.boolspocflbn:
                            gdat.phas, gdat.inptflbn, gdat.outp, gdat.legdoutp, gdat.tici, gdat.itoi = exopmain.retr_datatess(gdat.boolspocflbn) 
                        else:
                            gdat.time, gdat.inptraww, gdat.outp, gdat.legdoutp, gdat.tici, gdat.itoi = exopmain.retr_datatess(gdat.boolspocflbn)

                    if gdat.phastype == 'raww':
                        gdat.inpt = gdat.inptraww
        
                    if gdat.phastype == 'flbn':
                        if not gdat.boolspocflbn:   
                            strgsave = '%s_%d_%s_%04d_%04d_%04d' % \
                                            (datatype, np.log10(gdat.nois) + 5., gdat.zoomtype, gdat.numbphas, gdat.numbrele, gdat.numbirre)
                            pathsaveflbn = pathplot + 'save_flbn_%s' % strgsave + '.dat' 
                            pathsavephas = pathplot + 'save_phas_%s' % strgsave + '.dat' 
                            if not os.path.exists(pathsaveflbn):
                                cntr = 0
                                gdat.inptflbn = np.empty((gdat.numbdata, gdat.numbphas))
                                gdat.phas = np.empty((gdat.numbdata, gdat.numbphas))
                                # temp
                                flux_err = np.zeros(gdat.numbtime) + 1e-2
                                for k in gdat.indxdata:
                                    lcurobjt = lightkurve.lightcurve.LightCurve(flux=gdat.inptraww[k, :], time=gdat.time[k, :], \
                                                                                        flux_err=flux_err, time_format='jd', time_scale='utc')
                                    
                                    lcurobjtfold = lcurobjt.fold(gdat.peri[k])
                                    lcurobjtflbn = lcurobjtfold.bin(binsize=gdat.numbtime/gdat.numbphas, method='mean')
                                    gdat.inptflbn[k, :] = lcurobjtflbn.flux
                                    gdat.phas[k, :] = lcurobjtflbn.time
                                    assert np.isfinite(gdat.inptflbn[k, :]).all()

                                print 'Writing to %s...' % pathsaveflbn
                                np.savetxt(pathsaveflbn, gdat.inptflbn)
                                np.savetxt(pathsavephas, gdat.phas)
                            else:
                                print 'Reading from %s...' % pathsaveflbn
                                gdat.inptflbn = np.loadtxt(pathsaveflbn)
                                gdat.phas = np.loadtxt(pathsavephas)
                            gdat.inpt = gdat.inptflbn
                        else:
                            gdat.inpt = gdat.inptflbn

                    # plot
                    numbplotfram = 1
                    print 'Making plots of the input...'
                    listphastype = ['flbn']
                    if not gdat.boolspocflbn:
                        listphastype += ['raww']
                    for phastype in listphastype:
                        cntrplot = 0
                        for k in gdat.indxdata:
                            if k > 10:
                                break
                            if k % numbplotfram == 0:
                                figr, axis = plt.subplots(figsize=(12, 6))
                            if gdat.outp[k] == 1:
                                colr = 'b'
                            else:
                                colr = 'r'
                            if phastype == 'raww':
                                xdat = gdat.time[k, :]
                                ydat = gdat.inptraww[k, :]
                            if phastype == 'flbn':
                                xdat = gdat.phas[k, :]
                                ydat = gdat.inptflbn[k, :]
                            axis.plot(xdat, ydat, marker='o', markersize=5, alpha=0.6, color=colr, ls='')
                            if k % numbplotfram == 0 or k == gdat.numbdata - 1:
                                plt.tight_layout()
                                if phastype == 'raww':
                                    plt.xlabel('Time')
                                if phastype == 'flbn':
                                    plt.xlabel('Phase')
                                plt.ylabel('Flux')
                                plt.legend()
                                path = pathplot + 'inpt%s_%04d_%s_%04d_%04d' % (phastype, t, strgvarb, i, cntrplot) + '.png' 
                                print 'Writing to %s...' % path
                                plt.savefig(path)
                                plt.close()
                                cntrplot += 1
                    
                    #assert np.isfinite(gdat.inpt).all()
                    #assert np.isfinite(gdat.outp).all()

                    # divide the data set into training and test data sets
                    numbdatatest = int(gdat.fractest * gdat.numbdata)
                    gdat.inpttest = gdat.inpt[:numbdatatest]
                    gdat.outptest = gdat.outp[:numbdatatest]
                    gdat.inpttran = gdat.inpt[numbdatatest:]
                    gdat.outptran = gdat.outp[numbdatatest:]   

                    gdat.modl = Sequential()

                    # construct the neural net
                    # add a CNN
                    appdcon1(gdat)
                    
                    ## add the last output layer
                    appdfcon(gdat)
                    
                    gdat.modl.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
                    pathsave = pathplot + 'modlgrap_%s.png' % strgconf
                    keras.utils.plot_model(gdat.modl, to_file=pathsave)
                    
                    # temp -- this runs the central value redundantly and can be sped up by only running the central value once for all variables
                    # do the training for the specific value of the variable of interest
                    metr = retr_metr(gdat, i, strgvarb)

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

        if len(gdat.indxvalu[o]) == 1:
            continue

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
            if strgvarb == 'numbphas':
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
            
            if strgvarb in ['numbdata', 'numbphas', 'dept', 'nois', 'numbdimslayr', 'numbdatabtch']:
                axis.set_xscale('log')

            plt.legend()
            plt.tight_layout()

            plt.xlabel(labl)
            plt.ylabel(gdat.listlablmetr[l])

            path = pathplot + strgvarb + strgmetr + '.pdf' 
            plt.savefig(path)
            plt.close()
    

