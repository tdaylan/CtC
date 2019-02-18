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
import re

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


# hyperparams will go here for model optimization



# -----------------------------------------------------------------------------------


# locl and glob binning
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
        
        listdata = [inptloclfold, inptglobfold]

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
        localline = (np.arange(len(locl[k,:])), locl[k, :])
        globalline = (np.arange(len(glob[k,:])), glob[k, :])

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



def train_2inpt_model(model, epochs, locl, glob, labls, callbacks_list, disp=True):


    inptL1 = locl[:,:,None]
    inptG1 = glob[:,:,None]
    outp1 = labls


    hist = model.fit([inptL1, inptG1], outp1, epochs=epochs, validation_split=fractest, verbose=1, callbacks=callbacks_list)
    
    if disp:
        print(hist.summary())



# -----------------------------------------------------------------------------------


# WILL NEED TO INCLUDE HYPERPARAMETERS HERE TO DIFFERENTIATE MATRICES
pathsavematr = 'tess/matr/' + '{}'.format(str(modl.__name__)) + '.npy'

def matr_2inpt(model, locl, glob, labls):
    
    numbdatatest = fractest * len(locl)


    # separate the training from the testing
    inpttestL = locl[:numbdatatest, :]
    inpttranL = locl[numbdatatest:, :]

    inpttestG = glob[:numbdatatest, :]
    inpttranG = glob[numbdatatest:, :]

    outptest = labls[:numbdatatest]
    outptran = labls[numbdatatest:]


    numbepoc = len([name for name in os.listdir('tess/models/{}/'.format(str(model.__name__))) if os.path.isfile(name)])
    indxepoc = np.arange(numbepoc)


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


    for epoc in trange(indxepoc):

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


    listdata = [metr, conf_matr_vals]


    if 
        print ('Writing to %s...' % pathsavematr)
        objtfile = open(pathsavematr, 'wb')
        pickle.dump(listdata, objtfile, protocol=pickle.HIGHEST_PROTOCOL)
        objtfile.close()

    else:
        objtfile = open(pathsavematr, 'r')
        print ('Reading from %s...' % pathsavematr)
        listdata = pickle.load(objtfile)
        objtfile.close()
    

    return listdata


# -----------------------------------------------------------------------------------


def main():

    # data pull
    phases, fluxes, labels, _, _, _ = retr_datatess(True, boolplot=False)


    # local and global
    # this needs to get reformatted badly
    loclF, globF = gen_binned(fluxes, phases, phases)


    # sample relevance graphs
    inpt_before_train(loclF, globF, labels, save=False)

    
    # load the most previous weights from last training (could make this its own function)
    weights_list = [weights for weights in os.listdir(os.environ['EXOP_DATA_PATH'] + '/tess/models/') if weights.startswith(modl.__name__)]

    last_epoch = max(weights_list)

    prevMax = 0


    # initialize model
    model = modl()
    

    try:
        model.load_weights(last_epoch)
        print("Loading previous model's weights")

        
        regex = re.compile(r'\d+')
        prevMax = int(regex.findall(last_epoch)[0])

    except:
        print("Fresh, new model")


    
    modlpath = 'tess/models/{}/'.format(str(modl.__name__)) + 'weights-{}'.format(epoch + prevMax) + '.h5'
    checkpoint = ModelCheckpoint(modlpath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='max')
    tens_board = TensorBoard(log_dir='logs/{}/{}'.format(modl.__name__,time.time()))
    callbacks_list = [checkpoint, tens_board]  


    # need a conditional to check the shape of the model -- if two-input: use this function
    train_2inpt_model(model, numbepoc, loclF, globF, labels, callbacks_list)

    metr, conf = matr_2inpt(model, loclF, globF, labels)


# -----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()