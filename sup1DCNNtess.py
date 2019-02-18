import datetime, os, sys, argparse, random, pathlib, time

from tqdm import tqdm, trange
import numpy as np

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

import sklearn
from sklearn.metrics import confusion_matrix, roc_auc_score

import astropy as ap
import astropy.io.fits as fits

from scipy.interpolate import LSQUnivariateSpline

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import lightkurve

from exopmain import retr_datamock, retr_datatess 

from models import exonet, reduced

import pickle

"""
Need to figure out how to generate global and local views... then should be able to run...
"""


# -----------------------------------------------------------------------------------


# TO ALLOW EASY ACCESS TO SUBFOLDERS
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def disk_folder(path, *superpath, overwrite=False, home='EXOP_DATA_PATH'):
    """
    Very flimsy folder maker, works within scope needed for this project
    
    If folder needs to be within another folder, include superpath as the one-layer-up folder
    """

    if home == None:
        savePath = os.getcwd()
    else:
        savePath = os.environ[home]

    os.chdir(savePath)

    sup_indx = 0
    max_len = len(superpath)

    if max_len > 0:
        while sup_indx <= max_len:
            os.chdir(superpath[sup_indx])
            sup_indx += 1

    if not os.path.exists(path) or not overwrite:
        time.sleep(3)
        os.makedirs(path)
    
    os.chdir(savePath)


    printer = ''
    for sppath in superpath:
        printer += "/" + str(sppath)
    
    printer += "/" + str(path)

    print("Made folder {0} at {1}".format(path, printer))


# -----------------------------------------------------------------------------------


# TRAINING
fractest = 0.3

numbepoc = 20
indxepoc = np.arange(numbepoc)

datatype = 'tess'

modl = reduced

overwrite = False


# -----------------------------------------------------------------------------------

# FOR PRECISION AND RECALL

# points for thresholds graphing
points_thresh = 100
thresh = np.linspace(0.2, 0.9, points_thresh)


# -----------------------------------------------------------------------------------


# params for global vs local binning
"""
paperloclinpt = 200     # input shape from paper [local val]
papergloblinpt = 2000   # input shape from paper

loclbin = int(numbtime/paperloclinpt)
globbin = int(numbtime/papergloblinpt)

localtimebins = paperloclinpt
globaltimebins = papergloblinpt

localbinsindx = np.arange(localtimebins)
globalbinsindx = np.arange(globaltimebins)
"""


# -----------------------------------------------------------------------------------


# locl and glob binning
# TEMP NEED TO FIX _TIME INPUT 
def gen_binned(raw_flux, peri, _time, loclinptbins=200, globinptbins=2000, save=True):

    pathsave = os.environ['EXOP_DATA_PATH'] + '/tess/glob_locl.pickle'

    numbdata = len(raw_flux)
    indxdata = np.arange(numbdata)

    numbtime = len(_time)
    indxtime = np.arange(numbtime)

    loclbin = int(numbtime/loclinptbins)
    globbin = int(numbtime/globinptbins)

    if not os.path.exists(pathsave):
        # holder np arrays
        inptloclfold = np.empty((numbdata, loclinptbins))
        inptglobfold = np.empty((numbdata, globinptbins))

        # let our runner know what is happening :)
        print("\nGenerating binned")

        # for each curve in the inpt space
        for k in tqdm(indxdata):

            # BINNING ONLY WORKS FOR DATA THAT WE GENERATED :: MAKE MODULAR
            # also took out .flatten() as it was adding extra >>1 values in the data
            inptloclfold[k,:] = lightkurve.lightcurve.LightCurve(time=indxtime, flux=raw_flux[k,:], time_format='jd', time_scale='utc').fold(peri).bin(loclbin).flux # BIN hard coded (fix soon)
            inptglobfold[k,:] = lightkurve.lightcurve.LightCurve(time=indxtime, flux=raw_flux[k,:], time_format='jd', time_scale='utc').fold(peri).bin(globbin).flux # BIN hard coded
        
        listdata = [inptglobfold, inptloclfold]

        print ('Writing to %s...' % pathsave)
        objtfile = open(pathsave, 'wb')
        pickle.dump(listdata, objtfile, protocol=pickle.HIGHEST_PROTOCOL)
        objtfile.close()


    else:
        objtfile = open(pathsave, 'r')
        print ('Reading from %s...' % pathsave)
        listdata = pickle.load(objtfile)
        objtfile.close()

    return listdata



# -----------------------------------------------------------------------------------


# graphs inputs, and transformed inputs
def inpt_before_train(locl, glob, labl, num_graphs=10, save=True, overwrite=True):
    """
    Takes the binned data from gen_binned and makes input graphs
    Shows both the local and global view
    """

    indxdata = np.arange(len(labl))
 
    # gives just num_graphs # of plots
    indexer = int(len(indxdata)/num_graphs)
    indexes = indxdata[0::indexer]

    # let the user know what is happening :)
    print("\nMaking input graphs!")

    # let the user know with a progressbar :)
    # for just the 10 specified plots:
    for k in tqdm(indexes):

        # red is relevant (has an output that signifies a planet is present)
        if labl[k] == 1:
            colr = 'r'
            relevant = True
        # blue is irrelevant
        else:
            colr = 'b'
            relevant = False
        
        if relevant:
            textbox = 'Relevant'
        else:
            textbox = 'Irrelevant'

        # line for local, line for global [line is used liberally, just a collection of x and y points]
        # k indexes in for a SINGLE light curve
        localline = (localbinsindx, locl[k, :])
        globalline = (globalbinsindx, glob[k, :])

        # make the 2 subplots
        fig, axis = plt.subplots(2, 1, constrained_layout=True, figsize=(12,6))

        # plot 0 is the local plot, plot 1 is the global
        axis[0].plot(localline[0], localline[1], marker='o', alpha=0.6, color=colr)
        axis[1].plot(globalline[0], globalline[1], marker='o', alpha=0.6, color=colr)
        
        axis[0].set_title('Local ' + textbox)
        axis[1].set_title('Global ' + textbox)

        axis[0].set_xlabel('Timebins Index')
        axis[0].set_ylabel('Binned Flux')

        axis[1].set_xlabel('Timebins Index')
        axis[1].set_ylabel('Binned Flux')


        plt.tight_layout()

        if save:
            with cd(os.environ['EXOP_DATA_PATH'] + '/tess/plot/inptlabl/'):
                print ('Writing to %s...' % textbox+ '_{}'.format(k))
            
                plt.savefig(textbox+ '_{}'.format(k))
        
        else:
            plt.show()
                

    return None    


# -----------------------------------------------------------------------------------

modlpath = '{}_'.format(str(modl.__name__)) + 'weights' + '.h5'

pathsavemetr = 'metr_' + '{}_'.format(str(modl.__name__)) + '.npy'
pathsaveconf = 'conf_' + '{}_'.format(str(modl.__name__)) + '.npy'


matrdir = 'matrices'
# modldir = 'models'
# tb = 'tb_logs'

disk_folder(matrdir, (), overwrite=overwrite)
# disk_folder(modldir, (outpdir), overwrite=overwrite)
# disk_folder(tb, (outpdir, modldir), overwrite=overwrite)

checkpoint = ModelCheckpoint(modlpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
tens_board = TensorBoard(log_dir='logs/{}'.format(time.time()))
callbacks_list = [checkpoint, tens_board]

# THIS IS THE BOTTLENECK
def gen_metr(locl, glob, labl, numbdatatest):
    """
    This is the BOTTLENECK of this pipeline

    metr: prec, acc, recal; per epoch and threshold
    conf_matr_vals: trne, trpo, flne, flpo; per epoch and threshold

    the time is largely reliant on how many epochs are in indxepoc, and the size(or length) of the input data
    input data needs to be high enough to not overtrain over a small set, but not so big that the training never ends

    IDEALLY: there would be intermittent saving or updating, and the ability to start back up from where left off to
    use any of the data before it is completely finished
    """


    # load the files
    inptL = locl
    inptG = glob
    outp  = labl


    # with cd(modldir):
    # initialize model
    model = modl()

    try:
        model.load_weights(modlpath)
        print("Loading previous model's weights")
    except:
        print("New model")


    # initialize the matrix holding all metric values
    # INDEX 1: which epoch
    # INDEX 2: which threshold value is being tested against
    # INDEX 3: [IN THIS ORDER] precision, accuracy, recall (numerical values)
    # INDEX 4: train [0] test [1]
    metr = np.zeros((numbepoc, len(thresh), 3, 2)) - 1


    # we also want the confusion matrices for later use
    # INDEX 1: which epoch
    # INDEX 2: which threshold value is being tested against
    # INDEX 3: [IN THIS ORDER] trne, flpo, flne, trpo
    # INDEX 4: train [0] test [1]
    conf_matr_vals = np.zeros((numbepoc, len(thresh), 4, 2))


    # separate the training from the testing
    inpttestL = locl[:numbdatatest, :]
    inpttranL = locl[numbdatatest:, :]

    inpttestG = glob[:numbdatatest, :]
    inpttranG = glob[numbdatatest:, :]

    outptest = labl[:numbdatatest]
    outptran = labl[numbdatatest:]

    inptL1 = locl[:,:,None]
    inptG1 = glob[:,:,None]
    outp1 = labl

    # let our friends running this know what is happening
    print("\nGenerating Metric Matrix")

    # run through epochs
    for epoc in tqdm(indxepoc):
        
        hist = model.fit([inptL1, inptG1], outp1, epochs=1, validation_split=fractest, verbose=1, callbacks=callbacks_list)

        # NOTICE THAT THERE IS NO WAY TO TELL WHICH EPOCH YOU ARE ON IF LOADING FROM FILE
        # with cd(modldir):
            # model.save(modlpath)

        # train and then test 
        for i in trange(2):
            # i == 0 -> train
            # i == 1 -> test

            if i == 0:
                inptL = inpttranL
                inptG = inpttranG
                outp = outptran
                
                inptL = inptL[:, :, None]
                inptG = inptG[:, :, None]

                print('train')

                
            else:
                inptL = inpttestL
                inptG = inpttestG
                outp = outptest
                
                inptL = inptL[:, :, None]
                inptG = inptG[:, :, None]

                print('test')

            # only now, within the testing parameters, we test against a range of threshold values
            for threshold in trange(len(thresh)):

                outppred = (model.predict([inptL, inptG]) > thresh[threshold]).astype(int)

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

                if float(trpo+flpo+trne) != 0:
                    metr[epoc, threshold, 1] = float(trpo + trne)/(trpo + flpo + trne) # accuracy
                else:
                    pass

                if float(trpo + flne) > 0:
                    metr[epoc, threshold, 2] = trpo / float(trpo + flne) # recall
                else:
                    pass
                

        with cd(matrdir):
            
            if os.path.exists(pathsavemetr):
                dummy = np.load(pathsavemetr)
                print(dummy)
                concat = np.append(dummy, metr[epoc, :, :, :])
                np.save(pathsavemetr, concat)
                
            else:
                np.save(pathsavemetr, metr[epoc, :, :, :])


            if os.path.exists(pathsaveconf):
                dummy = np.load(pathsaveconf)
                concat = np.append(dummy, conf_matr_vals[epoc, :, :, :])
                np.save(pathsaveconf, concat)
                print(concat)
            else:
                np.save(pathsaveconf, conf_matr_vals[epoc, :, :, :])
   
    
    return None


# -----------------------------------------------------------------------------------


def main():

    phases, fluxes, labels, _, _, _ = retr_datatess(True, boolplot=False)

    nrow, ncol = fluxes.shape
    light_curves = np.reshape(fluxes, (nrow, ncol, 1))

    # colors for graphs
    colors =  ["r" if x else "b" for x in labels]

    # this needs to get reformatted badly
    loclF, globF = gen_binned(fluxes, phases)

    # sample relevance graphs
    inpt_before_train(loclF, globF, labels, save=False)

    numbdatatest = fractest * len(fluxes)



if __name__ == "__main__":
    main()