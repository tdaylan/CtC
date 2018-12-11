import datetime, os, sys, argparse, random
from progressbar import *
import numpy as np

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate, GlobalMaxPool1D

import tensorflow as tf

import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score

import astropy as ap

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import lightkurve

from exop import main as exopmain


widgets = ['Test: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()]





# -----------------------------------------------------------------------------------
# parameters

# training param
fractest = 0.3

numbepoc = 1
indxepoc = np.arange(numbepoc)

numbruns = 100

indxepoc = np.arange(numbepoc)
indxruns = np.arange(numbruns)


# mockdata param
datatype = 'here'

numbtime = 20000 # could maybe need to be 2001 to meet paper's specs

# find bin numb to get both 2001 and 201 with (we don't care about) remainders!

dept = 0.998
nois = 3e-3
numbdata = int(1e4)
fracplan = 0.2

numbplan = int(numbdata * fracplan)
numbnois = numbdata - numbplan

indxtime = np.arange(numbtime)
indxdata = np.arange(numbdata)

path_namer_dict = {'numbtime': numbtime, 'dept' : dept, 'nois' : nois, 'numbdata' : numbdata, 'fracplan':fracplan}
path_namer_str = ""
for key, value in path_namer_dict.items():
    paired = str(key) + str(value) + '_'
    path_namer_str += paired



# points for thresholds graphing
points_thresh = 100
thresh = [0.4 + i/(points_thresh*3) for i in range(points_thresh)]


# binning data
paperloclinpt = 200      # input shape from paper [local val]
papergloblinpt = 2000   # input shape from paper

localtimebins = paperloclinpt
globaltimebins = papergloblinpt

localbinsindx = np.arange(localtimebins)
globalbinsindx = np.arange(globaltimebins)



# names for folded .dat files
pathsavefoldLocl = 'savefold_%s_%s_%04dbins' % (datatype, 'locl', localtimebins) + path_namer_str +  '.dat'
pathsavefoldGlob = 'savefold_%s_%s_%04dbins' % (datatype, 'glob', globaltimebins) + path_namer_str + '.dat'
pathsavefoldoutp = 'savefold_%s_%s' % (datatype, 'outp') + path_namer_str + '.dat'

# name for pdf of inpt before running
inptb4path = 'inpt_'+path_namer_str+'.pdf'

pathsavemetr = 'metr_' + path_namer_str + '.npy'
# -----------------------------------------------------------------------------------

# CONVOLUTIONAL MODELS

# models from "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning"
# astronet
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

# 
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


    
    # print('shape test:', tf.keras.backend.shape(y))
    modldummy = Model(inputs=[localinput, globalinput], outputs=[y])
    print()
    modldummy.summary()

    print()

    # print('shape test:', tf.keras.backend.shape(x))
    modldummy = Model(inputs=[localinput, globalinput], outputs=[x])
    print()
    modldummy.summary()
    
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

# these need to have the same name but path has ()
# path = reduced()
modl = reduced

modlpath = 'reduced_' + path_namer_str + '.h5'
# -----------------------------------------------------------------------------------

# binning TOO SLOW
"""
def binn_lcur(numbtime, time, flux, peri, epoc, zoomtype='glob'):
    
    timefold = ((time - epoc) / peri + 0.25) % 1.
    
    if zoomtype == 'glob':
        minmtimefold = 0.
        maxmtimefold = 1.
    else:
        minmtimefold = 0.15
        maxmtimefold = 0.35
    binstimefold = np.linspace(minmtimefold, maxmtimefold, numbtime + 1)
    indxtime = np.arange(numbtime)
    fluxavgd = np.empty(numbtime)

    # print('\nfluxavgd before: \n', fluxavgd)

    for k in indxtime:
        indx = np.where((binstimefold[k] < timefold) & (timefold < binstimefold[k+1]))[0]
        fluxavgd[k] = np.mean(flux[indx])

    # print('\nfluxavgd after: \n', fluxavgd)
    
    # print(fluxavgd)
    return fluxavgd
"""
# -----------------------------------------------------------------------------------

#lcurobjt = lightkurve.lightcurve.LightCurve(flux=gdat.lcurdata[k], time=gdat.time[k], flux_err=flux_err, time_format='jd', time_scale='utc')
#lcurobjt.flatten()
#lcurobjt.fold(peri)
#lcurobjt.bin(numbtimebins)

# get the saved data
def gen_mockdata(datatype):
    
    pathname = path_namer_str

    if datatype == 'here':
        inptraww, outp, peri = exopmain.retr_datamock(numbplan=numbplan,\
                numbnois=numbnois, numbtime=numbtime, dept=dept, nois=nois)

        pathname += '_here.npz'
        np.savez(pathname, inptraww, outp, peri)

    elif datatype == 'ete6':
        time, inptraww, outp, tici, peri = exopmain.retr_dataete6(nois=nois, \
                                            numbdata=numbdata)
        
        pathname += '_ete6.npz'
        np.savez(pathname, time, inptraww, outp, tici, peri)
    
    return pathname

# bin and save
def gen_binned(path_namer, datatype):
    
    if datatype == 'here':
        loaded = np.load(path_namer+'_'+datatype+'.npz')

        inptraww, outp, peri = loaded['arr_0'], loaded['arr_1'], loaded['arr_2']
        # temp [hasn't been specified]
        time = np.arange(0,30)

    elif datatype == 'ete6':
        loaded = np.load(path_namer+'_'+datatype+'.npz')

        time, inptraww, outp, tici, peri = loaded['arr_0'], loaded['arr_1'], loaded['arr_2'], loaded['arr_3'], loaded['arr_4']

    inptloclfold = np.empty((numbdata,localtimebins))
    inptglobfold = np.empty((numbdata, globaltimebins))

    print("Generating binned")

    pbar = ProgressBar(widgets=widgets, maxval=len(indxdata))
    pbar.start()

    # cntr = 0
    for k in indxdata:
        # temp, dunno if this is right idea for peri [only uses first peri]
        # numbperi = peri[cntr].size
        # indxperi = np.arange(numbperi)

        tester = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc')

        # print(len(inptloclfold[k,:]), len(tester.flux))
        
        # only fold the planets by peri[k]
        if k < numbplan:
            peritemp = int(peri[k])
        else:
            peritemp = 10

        inptloclfold[k,:] = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc').flatten().fold(peritemp).bin(100).flux # BIN hard coded (fix soon)
        inptglobfold[k,:] = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc').flatten().fold(peritemp).bin(10).flux # BIN hard coded
        
        if k < 10:
            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            ax.plot(tester.time, tester.flux)
            ax.set_title('untouched')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('untouched{}'.format(k) + inptb4path)
            plt.close()

            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            ax.plot(tester.time, tester.flatten().flux)
            ax.set_title('Flattened')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('flattened{}'.format(k) + inptb4path)
            plt.close()            
            
            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            ax.plot(tester.time, tester.flatten().fold(peritemp).flux)
            ax.set_title('Folded')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('folded{}'.format(k) + inptb4path)
            plt.close()

            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            ax.plot(tester.time, tester.flatten().fold(peritemp).bin(10).flux)
            ax.set_title('Globally Folded')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('glob{}'.format(k) + inptb4path)
            plt.close()

            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            ax.plot(tester.time, tester.flatten().fold(peritemp).bin(100).flux)
            ax.set_title('Locally Folded')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('locl{}'.format(k) + inptb4path)
            plt.close()

        pbar.update(k)

    pbar.finish()

    np.savetxt(pathsavefoldLocl, inptloclfold)
    np.savetxt(pathsavefoldGlob, inptglobfold)
    np.savetxt(pathsavefoldoutp, outp)

    print('Writing local folded to %s...' % pathsavefoldLocl)
    print('Writing global folded to %s...' % pathsavefoldGlob)
    print('Writing output to %s...' % pathsavefoldoutp)

    return None


def inpt_before_train(locl, glob, outp, saveinpt=True):
    
    inptL = np.loadtxt(locl)
    inptG = np.loadtxt(glob)
    outp  = np.loadtxt(outp)
 
     

    # gives just 10 plots
    indexer = int(len(indxdata)/10)
    indexes = indxdata[0::indexer]

    print("Making input graphs!")

    pbar = ProgressBar(widgets=widgets, maxval=len(indexes))
    pbar.start()   
    pbarcounter = 0
    for k in indexes:
        if outp[k] == 1:
            colr = 'r'
        else:
            colr = 'b'
        
        localline = (localbinsindx, inptL[k, :])
        globalline = (globalbinsindx, inptG[k, :])

        
        fig, axis = plt.subplots(2, 1, constrained_layout=True, figsize=(12,6))


        axis[0].plot(localline[0], localline[1], marker='o', alpha=0.6, color=colr)
        axis[1].plot(globalline[0], globalline[1], marker='o', alpha=0.6, color=colr)
        
        axis[0].set_title('Local')
        axis[1].set_title('Global')

        axis[0].set_xlabel('Timebins Index')
        axis[0].set_ylabel('Binned Flux')

        axis[1].set_xlabel('Timebins Index')
        axis[1].set_ylabel('Binned Flux')

        
        plt.tight_layout()

        if saveinpt:
            plt.savefig('{}_'.format(k) + inptb4path)
            
        
        else:
            plt.show()
        
        pbar.update(pbarcounter)
        pbarcounter += 1
    pbar.finish()
    return None    
    
def gen_fitted_model(inptL, inptG, outp, model):
    
    if isinstance(inptL, str):
        inptL = np.loadtxt(inptL)
    if isinstance(inptG, str):
        inptG = np.loadtxt(inptG)
    if isinstance(outp, str):
        outp  = np.loadtxt(outp)

    modl = model()

    #got rid of: for y in indxepoc:
    inptfitL = inptL[:, :, None]
    inptfitG = inptG[:, :, None]
    
    hist = modl.fit([inptfitL, inptfitG], outp, epochs=numbepoc, validation_split=fractest, verbose=2)

    modl.save(modlpath)


    print(hist.history)

    return None

# NEEDS TO BE RERUN WITH PROPER THRESH VALS
def gen_metr(inptL, inptG, outp, fitmodel):

    if isinstance(inptL, str):
        inptL = np.loadtxt(inptL)
    if isinstance(inptG, str):
        inptG = np.loadtxt(inptG)
    if isinstance(outp, str):
        outp  = np.loadtxt(outp)

    if isinstance(fitmodel, str):
        # in case the fitmodel is being taken as the output from gen_fitted_model
        fitmodel = load_model(fitmodel)

    metr = np.zeros((numbepoc, len(thresh), 2, 3)) - 1

    numbdatatest = int(fractest * numbdata)

    inpttestL = inptL[:numbdatatest, :]
    inpttranL = inptL[numbdatatest:, :]

    inpttestG = inptG[:numbdatatest, :]
    inpttranG = inptG[numbdatatest:, :]

    outptest = outp[:numbdatatest]
    outptran = outp[numbdatatest:]

    print("Generating Metric Matrix")

    for epoc in indxepoc:
        

        for i in range(2):
            # i == 0 -> train
            # i == 1 -> test

            if i == 0:
                inptL = inpttranL
                inptG = inpttranG
                outp = outptran
                inptcol = 'train' 

            else:
                inptL = inpttestL
                inptG = inpttestG
                outp = outptest
                inptcol = 'test' 

            inptL = inptL[:, :, None]
            inptG = inptG[:, :, None]
            
            print("Epoch {0}, ".format(epoc) + inptcol)

            pbar = ProgressBar(widgets=widgets, maxval=len(thresh))
            pbar.start()

            for threshold in range(len(thresh)):

                precise, recalling = True, True

                outppred = (fitmodel.predict([inptL, inptG]) > thresh[threshold]).astype(int)

                matrconf = confusion_matrix(outp, outppred)

                if matrconf.size == 1:
                    matrconftemp = np.copy(matrconf)
                    matrconf = np.empty((2,2))
                    matrconf[0,0] = matrconftemp

                trne = matrconf[0,0]
                flpo = matrconf[0,1]
                flne = matrconf[1,0]
                trpo = matrconf[1,1]
                


                if float(trpo + flpo) > 0:
                    metr[epoc, threshold, i, 0] = trpo / float(trpo + flpo) # precision
                else:
                    precise = False

                metr[epoc, threshold, i, 1] = float(trpo + trne)/(trpo + flpo + trne) # accuracy

                if float(trpo + flne) > 0:
                    metr[epoc, threshold, i, 2] = trpo / float(trpo + flne) # recall
                else:
                    recalling = False
                """
                if not precise and not recalling:
                    pass
                    print('inptL')
                    summgene(inptL)
                    print('inptG')
                    summgene(inptG) 
                    print('outppred')
                    summgene(outppred)
                    print('confusion matrix\n', matrconf)
                
                else:
                    statement = 'viable'
                    

                if precise and recalling:
                    statement += ' and great!'

                try: 
                    print(statement)
                finally:
                    pass
                """
            pbar.finish()
    
    np.save(pathsavemetr, metr)
    
    return None

def summgene(varb):
    '''
    convenience function to quickly print a numpy array
    '''
    
    print ('min', np.amin(varb))
    print ('max', np.amax(varb))
    print ('mean', np.mean(varb))
    print ('shape', varb.shape)

def graph_PvR(inptL, inptG, outp, fitmodel, metr, saveinpt=True):
    
    if isinstance(inptL, str):
        inptL = np.loadtxt(inptL)
    if isinstance(inptG, str):
        inptG = np.loadtxt(inptG)
    if isinstance(outp, str):
        outp  = np.loadtxt(outp)

    if isinstance(fitmodel, str):
        # in case the fitmodel is being taken as the output from gen_fitted_model
        fitmodel = load_model(fitmodel)

    metr = np.load(pathsavemetr)


    inptL = inptL[:,:,None]
    inptG = inptG[:,:,None]
    y_pred = fitmodel.predict([inptL, inptG])

    y_real = outp

    try:
        auc = roc_auc_score(y_real, y_pred)
    except:
        print('y_pred is bad :(')
        auc = 0.

    textbox = '\n'.join((
        r'$\mathrm{Signal:Noise}=%.2f$' % (dept/nois, ),
        # r'$\mathrm{Gaussian Standard Deviation}=%.2f$' % (auc, ),
        r'$\mathrm{AUC}=%.8f$' % (auc, ),
        r'$\mathrm{Depth}=%.2f$' % (dept, )))
    


    x_points = []
    y_points = []


    for epoc in indxepoc:
        for i in range(2):
            
            if i == 0:
                typstr = 'test'

            else:
                typstr = 'train'


            print("Epoch: {0}, ".format(epoc) + typstr)

            pbar = ProgressBar(widgets=widgets, maxval=len(thresh))
            pbar.start()   

            for threshold in range(len(thresh)):
                
                x, y = metr[epoc, threshold, i, 2], metr[epoc, threshold, i, 0]

                if not np.isnan(x) and x != 0 and not np.isnan(y) and y != 0:
                    x_points.append(x) # recall
                    y_points.append(y) # precision

                pbar.update(threshold)
            pbar.finish()


    fig, axis = plt.subplots(constrained_layout=True, figsize=(12,6))
    axis.plot(x_points, y_points, marker='o', ls='', markersize=3, alpha=0.6)
    axis.axhline(1, alpha=.4)
    axis.axvline(1, alpha=.4)
    props = dict(boxstyle='round', alpha=0.4)
    axis.text(0.05, 0.25, textbox, transform=axis.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision v Recall')

    if saveinpt:
        plt.savefig('PvR{0}.pdf'.format(path_namer_str))
        plt.close()
    else:
        plt.show()


def graph_inpt_space(inptL, inptG, outp, fitmodel, metr, saveinpt=True):

    if isinstance(inptL, str):
        inptL = np.loadtxt(inptL)
    if isinstance(inptG, str):
        inptG = np.loadtxt(inptG)
    if isinstance(outp, str):
        outp  = np.loadtxt(outp)

    if isinstance(fitmodel, str):
        # in case the fitmodel is being taken as the output from gen_fitted_model
        fitmodel = load_model(fitmodel)

    metr = np.load(pathsavemetr)


    fig, axis = plt.subplots(2, 2, constrained_layout=True, figsize=(12,6))

    numbdatatest = int(fractest * numbdata)

    inpttestL = inptL[:numbdatatest, :]
    inpttranL = inptL[numbdatatest:, :]

    inpttestG = inptG[:numbdatatest, :]
    inpttranG = inptG[numbdatatest:, :]

    outptest = outp[:numbdatatest]
    outptran = outp[numbdatatest:]

    print("Graphing inpt based on conf_matr")
    pbar = ProgressBar(widgets=widgets, maxval=numbruns)
    pbar.start()  

    for run in range(numbruns):
        
        # for threshold in thresh:
        
        for i in range(2):
            # i == 0 : train
            # i == 1 : test
            
            if i == 0:
                inptL = inpttranL
                inptG = inpttranG
                outp = outptran
                col = 'r'
                
            else:
                inptL = inpttestL
                inptG = inpttestG
                outp = outptest
                col = 'b'
                
            inptL = inptL[:,:,None]
            inptG = inptG[:,:,None]
            
            outppred = (fitmodel.predict([inptL, inptG]) > 0.7).astype(int)
            
            matrconf = confusion_matrix(outp, outppred)
            
            if matrconf.size == 1:
                matrconftemp = np.copy(matrconf)
                matrconf = np.empty((2,2))
                matrconf[0,0] = matrconftemp
                
            
            trne = matrconf[0,0]
            flpo = matrconf[0,1]
            flne = matrconf[1,0]
            trpo = matrconf[1,1]



            axis[0,0].plot(run, trne, marker='o', ls='', markersize=3, alpha=0.1, color=col)
            axis[0,1].plot(run, flpo, marker='o', ls='', markersize=3, alpha=0.1, color=col)
            axis[1,0].plot(run, flne, marker='o', ls='', markersize=3, alpha=0.1, color=col)
            axis[1,1].plot(run, trne, marker='o', ls='', markersize=3, alpha=0.1, color=col)

        pbar.update(run)
    pbar.finish()
  
    axis[0,0].set_title('True Negative')
    axis[0,0].set_xlabel('Run #')
    axis[0,0].set_ylabel('Relative Flux')

    axis[0,1].set_title('False Positive')
    axis[0,1].set_xlabel('Run #')
    axis[0,1].set_ylabel('Relative Flux')

    axis[1,0].set_title('False Negative')
    axis[1,0].set_xlabel('Run #')
    axis[1,0].set_ylabel('Relative Flux')

    axis[1,1].set_title('True Positive')
    axis[1,1].set_xlabel('Run #')
    axis[1,1].set_ylabel('Relative Flux')

    plt.tight_layout()

    if saveinpt:
        plt.savefig('inptspace_confmatr' + inptb4path)
        plt.close()
    else:
        plt.show()




# --------------------------------------------------------------------------------

# script
mockdata = gen_mockdata(datatype)

if not os.path.exists(pathsavefoldLocl):
    gen_binned(path_namer_str, datatype)

inpt_before_train(pathsavefoldLocl, pathsavefoldGlob,pathsavefoldoutp)

if not os.path.exists(modlpath):
    gen_fitted_model(pathsavefoldLocl, pathsavefoldGlob,pathsavefoldoutp, modl)

if not os.path.exists(pathsavemetr):
    gen_metr(pathsavefoldLocl, pathsavefoldGlob, pathsavefoldoutp, modlpath)

graph_inpt_space(pathsavefoldLocl, pathsavefoldGlob, pathsavefoldoutp, modlpath, pathsavemetr, saveinpt=False)




