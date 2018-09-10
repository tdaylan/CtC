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

from gdatFile import gdatstrt


def summgene(varb):
    '''
    convenience function to quickly print a numpy array
    '''
    
    print (np.amin(varb))
    print (np.amax(varb))
    print (np.mean(varb))
    print (varb.shape)

# this can be wrapped in a function to allow for customization or 
# initialize the data here
gdat = gdatstrt()
# add a fully connected layer
gdat.addFcon()
# add a fully connected layer
gdat.addFcon()
# final output layer
gdat.addFcon(whchLayr='Last')

def expl( \
         # string indicating the model
         strguser='tansu', \
         strgtopo='fcon', \
         zoomtype='locl' """if local, operates normal, if local+globa or dub(double) it will take local and global at the same time""" \
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
    # functopo = globals().get(strgtopo)
    
    print ('CtC explorer initialized at %s.' % strgtimestmp)
    
    # path where plots will be generated
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
    
    # for each run
    for t in gdat.indxruns:
        
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
                
                gdat.dictmetr[strgvarb][0, 0, t, i] = metr[-1, 0, 0]
                gdat.dictmetr[strgvarb][1, 0, t, i] = metr[-1, 1, 0]
                gdat.dictmetr[strgvarb][0, 1, t, i] = metr[-1, 0, 1]
                gdat.dictmetr[strgvarb][1, 1, t, i] = metr[-1, 1, 1]
                gdat.dictmetr[strgvarb][0, 2, t, i] = metr[-1, 0, 2]
                gdat.dictmetr[strgvarb][1, 2, t, i] = metr[-1, 1, 2]
    
    alph = 0.5
    # plot the resulting metrics
    for o, strgvarb in enumerate(gdat.liststrgvarb): 
        for l, strgmetr in enumerate(gdat.liststrgmetr):
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
                    indx.append(np.where(gdat.dictmetr[strgvarb][r, l, :, i] != -1)[0])
                    ydat[i] = np.mean(gdat.dictmetr[strgvarb][r, l, :, indx[i]], axis=0)
                if len(indx) > 0:
                    for i in gdat.indxvalu[o]:
                        yerr[0, i] = ydat[i] - np.percentile(gdat.dictmetr[strgvarb][r, l, indx[i], i], 5.)
                        yerr[1, i] = np.percentile(gdat.dictmetr[strgvarb][r, l, :, i], 95.) - ydat[i]
                
                    if r == 0:
                        linestyl = '--' # unused
                    else:
                        linestyl = ''
                
                temp, listcaps, temp = axis.errorbar(gdat.listvalu[strgvarb], ydat, yerr=yerr, label=gdat.listlablrtyp[r], capsize=10, marker='o', \
                                                                                    ls='', markersize=10, lw=3, alpha=alph, color=colr)
                for caps in listcaps:
                    caps.set_markeredgewidth(3)
            
                for t in gdat.indxruns:
                    axis.plot(gdat.listvalu[strgvarb], gdat.dictmetr[strgvarb][r, l, t, :], marker='D', ls='', markersize=5, alpha=alph, color=colr)
            
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
        
            axis.set_ylabel(gdat.listlablmetr[l]) 
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

