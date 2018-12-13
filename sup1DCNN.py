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
import astropy.io.fits as fits

from scipy.interpolate import LSQUnivariateSpline

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import lightkurve

from exop import main as exopmain


widgets = ['Working! ', Percentage(), ' ', Bar(marker='#',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()]

""" 
color convention:

true pos: blue
true neg: green
fals pos: red
fals neg: orange

write disposition (as a text box to the top of the graph) [for the 2x2 trflnegpos graphs]
write relevant vs irrelevant for the input curves

write tmag and rp and period and transit depth and src in a text box

SRC********* need to keep the two seperate


FLATTEN*****
    need to mask the transit before flattening
    this requires knowing where the transit is!
    look for examples in lightkurve

    Tansu's impression:
        want get the period (read from one of the columns of the csv)
        use that period to mask the transit
        HOW TO DO THAT: 
            no immediate answer on lightkurve...


            try this instead of flattening:


            detrending with masking
            to do the detrending itself
            scipt.inerpolate.LSQUnivariateSpline(x,y,t)
                x: time
                y: relative flux
                t: linspace(min(x), max(x) [excluding these values], of length 10-15)

                returns fit to these points

            subtract lightcurve - LSQFit

            ** or **

            to get the mask, without the phase information:

                replace the 'transit location' in the mask with NaNs so we can subtract this


using datavalidation objects, so we cannot just read the lightcurve file in lightkurve

so we just send the flux and time and pass them to the lightkurve arguments as we did before 

download the datavalidationfiles using shell script
within for loop:
    pick out file names using os.listdir 
    fnmatch to filter out file names

    ^^ set to path name

    

path = 
"""




# -----------------------------------------------------------------------------------
# parameters

# training param
fractest = 0.3

numbepoc = 20
indxepoc = np.arange(numbepoc)

indxepoc = np.arange(numbepoc)


# mockdata param
datatype = 'here'

numbtime = 20000 # could maybe need to be 2001 to meet paper's specs

# find bin numb to get both 2001 and 201 with (we don't care about) remainders!

dept = 0.998
nois = 3e-3
numbdata = int(2e4)
fracplan = 0.35

numbplan = int(numbdata * fracplan)
numbnois = numbdata - numbplan

indxtime = np.arange(numbtime)
indxdata = np.arange(numbdata)

numbdatatest = int(fractest * numbdata)

# ONLY FOR MOCK DATA, SHOULD RENAME THINGS FOR OTHER [TESS] INPUTS
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

loclbin = int(numbtime/paperloclinpt)
globbin = int(numbtime/papergloblinpt)

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
pathsaveconf = 'conf_' + path_namer_str + '.npy'
# -----------------------------------------------------------------------------------

# CONVOLUTIONAL MODELS

# models from "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning"
# aka astronet
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

modlpath = '{}_'.format(str(modl.__name__)) + path_namer_str + '.h5'
# -----------------------------------------------------------------------------------

# get the saved data
def gen_mockdata(datatype):
    """
    Pretty straightforward: datatype is a string
    
    Ex:
    'here' : mockdata generated in exopmain;
    'ete6' : data from ete6 (still pulled from exopmain);
    'tess' : data from TESS (pulled from exopmain);

    Saves the input data as a .npz file
    
    Returns the final pathname (so if needed you can print, or assign to variable)
    """

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
    """
    Takes the input data (as the .npz file from gen_mockdata) and bins the data
    

    Uses lightkurve to flatten, fold, and bin the data


    [Not modular YET] Currently will make 10 sets of graphs of each input space, with each set containing the raw lightcurve,
    the flattened, the folded, and finally the binned curves. 


    Current state:  

    removed flattening as it caused the transits to display a weird 'before and after upwards peak'

    binning is hard coded above to ensure size is correct for the models being trained


    Returns None

    Saves local, global, and output to three separate .dat files 
    """


    # load data based on type
    if datatype == 'here':
        loaded = np.load(path_namer+'_'+datatype+'.npz')

        inptraww, outp, peri = loaded['arr_0'], loaded['arr_1'], loaded['arr_2']
        # temp [hasn't been specified]
        time = np.arange(0,30)

    elif datatype == 'ete6':
        loaded = np.load(path_namer+'_'+datatype+'.npz')

        time, inptraww, outp, tici, peri = loaded['arr_0'], loaded['arr_1'], loaded['arr_2'], loaded['arr_3'], loaded['arr_4']

    # holder np arrays
    inptloclfold = np.empty((numbdata,localtimebins))
    inptglobfold = np.empty((numbdata, globaltimebins))

    # let our runner know what is happening :)
    print("\nGenerating binned")

    # further let the runner know what is happening, by running a progress bar :))
    pbar = ProgressBar(widgets=widgets, maxval=len(indxdata))
    pbar.start()

    # for each curve in the inpt space
    for k in indxdata:

        # for datatype 'here' the period of the planets are found in the indeces k < numbplan (the planets are the first numbplan# of light curves)
        if k < numbplan:
            peritemp = int(peri[k])
        else:
            peritemp = 10

        # temp! removed flatten before fold
        inptloclfold[k,:] = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc').fold(peritemp).bin(loclbin).flux # BIN hard coded (fix soon)
        inptglobfold[k,:] = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc').fold(peritemp).bin(globbin).flux # BIN hard coded
        
        # [HARD CODED, MAKE MODULAR] graphs showing transformations on input space
        if k < 10:

            # a lightkurve object purely for graphing purposes
            tester = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc')

            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            ax.plot(tester.time, tester.flux)
            ax.set_title('untouched')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('untouched{}'.format(k) + inptb4path)
            plt.close()

            """
            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            ax.plot(tester.time, tester.flatten().flux)
            ax.set_title('Flattened')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('flattened{}'.format(k) + inptb4path)
            plt.close()            
            """

            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            ax.plot(tester.time, tester.fold(peritemp).flux)
            ax.set_title('Folded')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('folded{}'.format(k) + inptb4path)
            plt.close()

            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            temp = tester.fold(peritemp).bin(globbin) 
            ax.plot(temp.time, temp.flux) 
            ax.set_title('Globally Folded')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('glob{}'.format(k) + inptb4path)
            plt.close()

            fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
            temp = tester.fold(peritemp).bin(loclbin) 
            ax.plot(temp.time, temp.flux) 
            ax.set_title('Locally Folded')
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            plt.tight_layout()
            plt.savefig('locl{}'.format(k) + inptb4path)
            plt.close()

        pbar.update(k)

    pbar.finish()

    # save the data generated
    np.savetxt(pathsavefoldLocl, inptloclfold)
    np.savetxt(pathsavefoldGlob, inptglobfold)
    np.savetxt(pathsavefoldoutp, outp)

    # let the user know we are done here :)
    print('Writing local folded to %s...' % pathsavefoldLocl)
    print('Writing global folded to %s...' % pathsavefoldGlob)
    print('Writing output to %s...' % pathsavefoldoutp)

    return None

# graphs inputs, and transformed inputs
def inpt_before_train(locl, glob, outp, saveinpt=True):
    """
    Takes the binned data from gen_binned and makes input graphs

    Currently hard-coded to make exactly 10

    Shows both the local and global view
    """

    # load the files
    inptL = np.loadtxt(locl)
    inptG = np.loadtxt(glob)
    outp  = np.loadtxt(outp)
 
    # gives just 10 plots
    indexer = int(len(indxdata)/10)
    indexes = indxdata[0::indexer]

    # let the user know what is happening :)
    print("\nMaking input graphs!")

    # let the user know with a progressbar :)
    pbar = ProgressBar(widgets=widgets, maxval=len(indexes))
    pbar.start()  

    # for the progressbar
    pbarcounter = 0

    # for just the 10 specified plots:
    for k in indexes:

        # red is relevant (has an output that signifies a planet is present)
        if outp[k] == 1:
            colr = 'r'
        # blue is irrelevant
        else:
            colr = 'b'
        
        # line for local, line for global [line is used liberally, just a collection of x and y points]
        # k indexes in for a SINGLE light curve
        localline = (localbinsindx, inptL[k, :])
        globalline = (globalbinsindx, inptG[k, :])

        # make the 2 subplots
        fig, axis = plt.subplots(2, 1, constrained_layout=True, figsize=(12,6))

        # plot 1 is the local plot, plot 2 is the global
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

# single-epoch-trained model    
def gen_fitted_model(inptL, inptG, outp, model):
    """
    Initializes a single-epoch-trained model just to be called at later points

    Arguably, this might be entirely useless function that takes up more space than it is worth,
    depends on how much having a saved, barely trained function is worth
    """

    # inputs should be strings of the filenames of the data
    if isinstance(inptL, str):
        inptL = np.loadtxt(inptL)
    if isinstance(inptG, str):
        inptG = np.loadtxt(inptG)
    if isinstance(outp, str):
        outp  = np.loadtxt(outp)

    # initialize model
    modl = model()

    # this is TRAINING so we only use the training data
    inptL = inptL[numbdatatest:, :]
    inptG = inptG[numbdatatest:, :]
    outp = outp[numbdatatest:]

    # make the size of the inputs fit the model
    inptfitL = inptL[:, :, None]
    inptfitG = inptG[:, :, None]
    
    # fit the model for the first epoch (this is largely to just have a baseline model to keep training later)
    hist = modl.fit([inptfitL, inptfitG], outp, epochs=1, validation_split=fractest, verbose=2)

    # save for later use
    modl.save(modlpath)

    # optional, shows how well fit
    print(hist.history)

    return None

# THIS IS THE VALUABLE MATRIX DATA in matrix metr and conf_matr_vals
def gen_metr(inptL, inptG, outp, fitmodel):
    """
    This is the BOTTLENECK of this pipeline

    metr: prec, acc, recal; per epoch and threshold
    conf_matr_vals: trne, trpo, flne, flpo; per epoch and threshold

    the time is largely reliant on how many epochs are in indxepoc, and the size(or length) of the input data
    input data needs to be high enough to not overtrain over a small set, but not so big that the training never ends

    IDEALLY: there would be intermittent saving or updating, and the ability to start back up from where left off to
    use any of the data before it is completely finished
    """

    # inputs should be strings, loads inputs from files
    if isinstance(inptL, str):
        inptL = np.loadtxt(inptL)
    if isinstance(inptG, str):
        inptG = np.loadtxt(inptG)
    if isinstance(outp, str):
        outp  = np.loadtxt(outp)
    if isinstance(fitmodel, str):
        fitmodel = load_model(fitmodel)


    # initialize the matrix holding all metric values
    # INDEX 1: which epoch
    # INDEX 2: which threshold value is being tested against
    # INDEX 3: [IN THIS ORDER] precision, accuracy, recall (numerical values)
    metr = np.zeros((numbepoc, len(thresh), 3)) - 1


    # we also want the confusion matrices for later use
    # INDEX 1: which epoch
    # INDEX 2: which threshold value is being tested against
    # INDEX 3: [IN THIS ORDER] trne, flpo, flne, trpo
    conf_matr_vals = np.zeros((numbepoc, len(thresh), 4))


    # separate the training from the testing
    inpttestL = inptL[:numbdatatest, :]
    inpttranL = inptL[numbdatatest:, :]

    inpttestG = inptG[:numbdatatest, :]
    inpttranG = inptG[numbdatatest:, :]

    outptest = outp[:numbdatatest]
    outptran = outp[numbdatatest:]

    # let our friends running this know what is happening
    print("\nGenerating Metric Matrix")

    # run through epochs
    for epoc in indxepoc:
        
        # train and then test (we shouldn't run the metric return on the training data)
        for i in range(2):
            # i == 0 -> train
            # i == 1 -> test

            if i == 0:
                inptL = inpttranL
                inptG = inpttranG
                outp = outptran
                
                inptL = inptL[:, :, None]
                inptG = inptG[:, :, None]

                # we loaded in an already once trained model, so to keep with our notation, we should exclude epoc 1
                if epoc > 1:
                    hist = fitmodel.fit([inptL, inptG], outp, epochs=1, validation_split=fractest, verbose=2)
                else:
                    pass
                
            else:
                inptL = inpttestL
                inptG = inpttestG
                outp = outptest
                
                inptL = inptL[:, :, None]
                inptG = inptG[:, :, None]

                # only now, within the testing parameters, we test against a range of threshold values
                for threshold in range(len(thresh)):

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
                    
                    # update conf matrix holder
                    conf_matr_vals[epoc, threshold, 0] = trne
                    conf_matr_vals[epoc, threshold, 1] = flpo
                    conf_matr_vals[epoc, threshold, 2] = flne
                    conf_matr_vals[epoc, threshold, 3] = trpo


                    # update metr with only viable data (test for positive values)
                    if float(trpo + flpo) > 0:
                        metr[epoc, threshold, 0] = trpo / float(trpo + flpo) # precision
                    else:
                        pass

                    metr[epoc, threshold, 1] = float(trpo + trne)/(trpo + flpo + trne) # accuracy

                    if float(trpo + flne) > 0:
                        metr[epoc, threshold, 2] = trpo / float(trpo + flne) # recall
                    else:
                        pass

    fitmodel.save('trained_' + modlpath)
    np.save(pathsavemetr, metr)
    np.save(pathsaveconf, conf_matr_vals)
    
    return None

# graphs prec vs recal 
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
        r'$\mathrm{Depth}=%.4f$' % (dept, )))
    


    x_points = []
    y_points = []

    fig, axis = plt.subplots(constrained_layout=True, figsize=(12,6))

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
                    axis.plot(x, y, marker='o', ls='', markersize=3, alpha=0.6)

                pbar.update(threshold)
            pbar.finish()


    
    
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

# graphs conf matrix per epoch
# remake to just take conf_matr_vals
def graph_conf(inptL, inptG, outp, fitmodel, metr, saveinpt=True):

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

    

    inpttestL = inptL[:numbdatatest, :]
    inpttranL = inptL[numbdatatest:, :]

    inpttestG = inptG[:numbdatatest, :]
    inpttranG = inptG[numbdatatest:, :]

    outptest = outp[:numbdatatest]
    outptran = outp[numbdatatest:]

    print("Graphing inpt based on conf_matr") 

    for i in range(2):
        # i == 0 : train
        # i == 1 : test

        for epoch in range(10):
            
            print("\nEpoch ", epoch)

            
            if i == 0:
                inptL = inpttranL
                inptG = inpttranG
                outp = outptran
                shape = 'o'
                
            else:
                inptL = inpttestL
                inptG = inpttestG
                outp = outptest
                shape = 'v'
                
            inptL = inptL[:,:,None]
            inptG = inptG[:,:,None]
            

            hist = fitmodel.fit([inptL, inptG], outp, epochs=1, validation_split=fractest, verbose=1)
            
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


            
            axis[0,0].plot(epoch, trne, marker=shape, ls='', markersize=3, alpha=0.3, color='g')
            axis[0,1].plot(epoch, flpo, marker=shape, ls='', markersize=3, alpha=0.3, color='r')
            axis[1,0].plot(epoch, flne, marker=shape, ls='', markersize=3, alpha=0.3, color='#FFA500') # this is the color orange
            axis[1,1].plot(epoch, trpo, marker=shape, ls='', markersize=3, alpha=0.3, color='b')

  
    axis[0,0].set_title('True Negative')
    axis[0,0].set_xlabel('Epoch')
    axis[0,0].set_ylabel('True Negative')

    axis[0,1].set_title('False Positive')
    axis[0,1].set_xlabel('Epoch')
    axis[0,1].set_ylabel('False Positive')

    axis[1,0].set_title('False Negative')
    axis[1,0].set_xlabel('Epoch')
    axis[1,0].set_ylabel('False Negative')

    axis[1,1].set_title('True Positive')
    axis[1,1].set_xlabel('Epoch')
    axis[1,1].set_ylabel('True Positive')

    plt.tight_layout()

    if saveinpt:
        plt.savefig('inptspace_confmatr' + inptb4path)
        plt.close()
    else:
        plt.show()




# --------------------------------------------------------------------------------

# script
# mockdata = gen_mockdata(datatype)

if not os.path.exists(pathsavefoldLocl):
    gen_binned(path_namer_str, datatype)

inpt_before_train(pathsavefoldLocl, pathsavefoldGlob,pathsavefoldoutp)

if not os.path.exists(modlpath):
    gen_fitted_model(pathsavefoldLocl, pathsavefoldGlob,pathsavefoldoutp, modl)

if not os.path.exists(pathsavemetr):
    gen_metr(pathsavefoldLocl, pathsavefoldGlob, pathsavefoldoutp, modlpath)

graph_conf(pathsavefoldLocl, pathsavefoldGlob, pathsavefoldoutp, modlpath, pathsavemetr)

graph_PvR(pathsavefoldLocl, pathsavefoldGlob, pathsavefoldoutp, modlpath, pathsavemetr)



