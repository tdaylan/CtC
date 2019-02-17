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


def disk_folder(path, *superpath, overwrite=False, home=None):
    """
    Very flimsy folder maker, works within scope needed for this project
    
    If folder needs to be within another folder, include superpath as the one-layer-up folder
    """

    if home == None:
        savePath = os.getcwd()
    else:
        savePath = home

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

# FOR DATA GENERATION/LOADING INTO DISK

# external folder
outpdir = 'outputs_' + datatype + '_' + modl.__name__
# make the external folder
disk_folder(outpdir, (), overwrite=overwrite)

datadir = 'generated_data'

disk_folder(datadir, (outpdir), overwrite=overwrite)


# 'here' params ----------------------------------------------------------------------
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

# name the files accordingly
path_namer_dict = {'numbtime': numbtime, 'dept' : dept, 'nois' : nois, 'numbdata' : numbdata, 'fracplan':fracplan}
path_namer_str = ""
for key, value in path_namer_dict.items():
    paired = str(key) + str(value) + '_'
    path_namer_str += paired


datafile = path_namer_str+'_{}.npz'.format(datatype)
# ------------------------------------------------------------------------------------


def load_start():
    with cd(outpdir):
        with cd(datadir):
            if datatype == 'here':

                if not os.path.exists(datafile) and not overwrite:

                    inptraww, outp, peri = retr_datamock(numbplan=numbplan,\
                            numbnois=numbnois, numbtime=numbtime, dept=dept, nois=nois)

                    np.savez(datafile, inptraww, outp, peri)


            elif datatype == 'ete6':
                pass


            elif datatype == 'tess':
                pass

 
# -----------------------------------------------------------------------------------

# BINNING


# params --------------------------------------------------------
paperloclinpt = 200     # input shape from paper [local val]
papergloblinpt = 2000   # input shape from paper

loclbin = int(numbtime/paperloclinpt)
globbin = int(numbtime/papergloblinpt)

localtimebins = paperloclinpt
globaltimebins = papergloblinpt

localbinsindx = np.arange(localtimebins)
globalbinsindx = np.arange(globaltimebins)
# ---------------------------------------------------------------

# saving data

# names for folded .dat files
pathsavefoldLocl = 'savefold_%s_%s_%04dbins' % (datatype, 'locl', localtimebins) + path_namer_str +  '.dat'
pathsavefoldGlob = 'savefold_%s_%s_%04dbins' % (datatype, 'glob', globaltimebins) + path_namer_str + '.dat'
pathsavefoldoutp = 'savefold_%s_%s' % (datatype, 'outp') + path_namer_str + '.dat'

# name for pdf of inpt before running
inptb4path = 'inpt_'+path_namer_str+'.pdf'


binndir = 'binned_data'
binnimgdir = 'lightcurves'


disk_folder(binndir, (outpdir), overwrite=overwrite)
disk_folder(binnimgdir, (outpdir, binndir), overwrite=overwrite)

# run
def gen_binned():


    with cd(outpdir):
        with cd(datadir):

            loaded_data = np.load(datafile)

            if datatype == 'here':
                inptraww, outp, peri = loaded_data['arr_0'], loaded_data['arr_1'], loaded_data['arr_2']


            elif datatype =='ete6':
                time, inptraww, outp, tici, peri = loaded_data['arr_0'], loaded_data['arr_1'], loaded_data['arr_2'], loaded_data['arr_3'], loaded_data['arr_4']


            elif datatype == 'tess':
                pass

        with cd(binndir):

            # holder np arrays
            inptloclfold = np.empty((numbdata,localtimebins))
            inptglobfold = np.empty((numbdata, globaltimebins))

            # let our runner know what is happening :)
            print("\nGenerating binned")

            # further let the runner know what is happening, by running a progress bar :))
            # using tqdm as progress bar!

            # for each curve in the inpt space
            for k in tqdm(indxdata):

                # for datatype 'here' the period of the planets are found in the indeces k < numbplan (the planets are the first numbplan# of light curves)
                if datatype == 'here':
                    if k < numbplan:
                        peritemp = int(peri[k])
                    else:
                        peritemp = 10
                
                elif datatype == 'ete6':
                    pass
                
                elif datatype == 'tess':
                    pass

                # BINNING ONLY WORKS FOR DATA THAT WE GENERATED :: MAKE MODULAR
                # also took out .flatten() as it was adding extra >>1 values in the data
                inptloclfold[k,:] = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc').fold(peritemp).bin(loclbin).flux # BIN hard coded (fix soon)
                inptglobfold[k,:] = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc').fold(peritemp).bin(globbin).flux # BIN hard coded
                
                # [HARD CODED, MAKE MODULAR] graphs showing transformations on input space
                if k < 10:
                    with cd(binnimgdir):
                        # a lightkurve object purely for graphing purposes
                        tester = lightkurve.lightcurve.LightCurve(time=indxtime, flux=inptraww[k,:], time_format='jd', time_scale='utc')

                        fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(12,6))
                        ax[0,0].plot(tester.time, tester.flux)
                        ax[0,0].set_title('untouched')
                        ax[0,0].set_xlabel('Time')
                        ax[0,0].set_ylabel('Flux')
                        # plt.tight_layout()
                        # plt.savefig('untouched{}'.format(k) + inptb4path)
                        # plt.close()

                        
                        # fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
                        # ax[1].plot(tester.time, tester.flatten().flux)
                        # ax[1].set_title('Flattened')
                        # ax[1].set_xlabel('Time')
                        # ax[1].set_ylabel('Flux')
                        # plt.tight_layout()
                        # plt.savefig('flattened{}'.format(k) + inptb4path)
                        # plt.close()            
                        

                        # fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
                        ax[0,1].plot(tester.time, tester.fold(peritemp).flux)
                        ax[0,1].set_title('Folded')
                        ax[0,1].set_xlabel('Time')
                        ax[0,1].set_ylabel('Flux')
                        # plt.tight_layout()
                        # plt.savefig('folded{}'.format(k) + inptb4path)
                        # plt.close()

                        # fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
                        temp = tester.fold(peritemp).bin(globbin) 
                        ax[1,0].plot(temp.time, temp.flux) 
                        ax[1,0].set_title('Globally Binned')
                        ax[1,0].set_xlabel('Time')
                        ax[1,0].set_ylabel('Flux')
                        # plt.tight_layout()
                        # plt.savefig('globl{}'.format(k) + inptb4path)
                        # plt.close()

                        # fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))
                        temp = tester.fold(peritemp).bin(loclbin) 
                        ax[1,1].plot(temp.time, temp.flux) 
                        ax[1,1].set_title('Locally Binned')
                        ax[1,1].set_xlabel('Time')
                        ax[1,1].set_ylabel('Flux')
                        # plt.tight_layout()
                        # plt.savefig('locl{}'.format(k) + inptb4path)
                        # plt.close()

                        plt.tight_layout()
                        plt.savefig('{}'.format(k) + inptb4path)
                        plt.close()


            # save the data generated
            np.savetxt(pathsavefoldLocl, inptloclfold)
            np.savetxt(pathsavefoldGlob, inptglobfold)
            np.savetxt(pathsavefoldoutp, outp)

            # let the user know we are done here :)
            print('Writing local folded to %s...' % pathsavefoldLocl)
            print('Writing global folded to %s...' % pathsavefoldGlob)
            print('Writing output to %s...' % pathsavefoldoutp)
    return None


# -----------------------------------------------------------------------------------

inptdir = 'input_images'

disk_folder(inptdir, (outpdir), overwrite=overwrite)


# graphs inputs, and transformed inputs
def inpt_before_train(num_graphs=10, save=True, overwrite=True):
    """
    Takes the binned data from gen_binned and makes input graphs
    Shows both the local and global view
    """

    with cd(outpdir):
        with cd(binndir):

            # load the files
            inptL = np.loadtxt(pathsavefoldLocl)
            inptG = np.loadtxt(pathsavefoldGlob)
            outp  = np.loadtxt(pathsavefoldoutp)
 
        # gives just num_graphs # of plots
        indexer = int(len(indxdata)/num_graphs)
        indexes = indxdata[0::indexer]

        # let the user know what is happening :)
        print("\nMaking input graphs!")

        # let the user know with a progressbar :)
        # for just the 10 specified plots:
        for k in tqdm(indexes):

            # red is relevant (has an output that signifies a planet is present)
            if outp[k] == 1:
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
            localline = (localbinsindx, inptL[k, :])
            globalline = (globalbinsindx, inptG[k, :])

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

            with cd(inptdir):
                plt.savefig(textbox+ '_{}_'.format(k) + inptb4path)
                

    return None    


# -----------------------------------------------------------------------------------

modlpath = '{}_'.format(str(modl.__name__)) + path_namer_str + '.h5'

pathsavemetr = 'metr_' + path_namer_str + '.npy'
pathsaveconf = 'conf_' + path_namer_str + '.npy'


matrdir = 'matrices'
# modldir = 'models'
# tb = 'tb_logs'

disk_folder(matrdir, (outpdir), overwrite=overwrite)
# disk_folder(modldir, (outpdir), overwrite=overwrite)
# disk_folder(tb, (outpdir, modldir), overwrite=overwrite)

checkpoint = ModelCheckpoint(modlpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
tens_board = TensorBoard(log_dir='logs/{}'.format(time()))
callbacks_list = [checkpoint, tens_board]

# THIS IS THE BOTTLENECK
def gen_metr():
    """
    This is the BOTTLENECK of this pipeline

    metr: prec, acc, recal; per epoch and threshold
    conf_matr_vals: trne, trpo, flne, flpo; per epoch and threshold

    the time is largely reliant on how many epochs are in indxepoc, and the size(or length) of the input data
    input data needs to be high enough to not overtrain over a small set, but not so big that the training never ends

    IDEALLY: there would be intermittent saving or updating, and the ability to start back up from where left off to
    use any of the data before it is completely finished
    """

    with cd(outpdir):
        with cd(binndir):

            # load the files
            inptL = np.loadtxt(pathsavefoldLocl)
            inptG = np.loadtxt(pathsavefoldGlob)
            outp  = np.loadtxt(pathsavefoldoutp)


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
        inpttestL = inptL[:numbdatatest, :]
        inpttranL = inptL[numbdatatest:, :]

        inpttestG = inptG[:numbdatatest, :]
        inpttranG = inptG[numbdatatest:, :]

        outptest = outp[:numbdatatest]
        outptran = outp[numbdatatest:]

        inptL1 = inptL[:,:,None]
        inptG1 = inptG[:,:,None]
        outp1 = outp

        # let our friends running this know what is happening
        print("\nGenerating Metric Matrix")

        # run through epochs
        for epoc in tqdm(indxepoc):
            
            hist = model.fit([inptL1, inptG1], outp1, epochs=1, validation_split=fractest, verbose=1, callbacks=callbacks_list)

            # NOTICE THAT THERE IS NO WAY TO TELL WHICH EPOCH YOU ARE ON IF LOADING FROM FILE
            # with cd(modldir):
                # model.save(modlpath)

            # train and then test 
            for i in range(2):
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
                        metr[epoc, threshold, 1] = 0

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

pvrdir = 'Precision_vs_Recall'

disk_folder(pvrdir, (outpdir), overwrite=overwrite)


# graphs prec vs recal 
def graph_PvR():
    
    with cd(outpdir):
        with cd(binndir):

            # load the files
            inptL = np.loadtxt(pathsavefoldLocl)
            inptG = np.loadtxt(pathsavefoldGlob)
            outp  = np.loadtxt(pathsavefoldoutp)

        model = modl()
        model.load_weights(modlpath)

        with cd(matrdir):
            metr = np.load(pathsavemetr)

    # TEMP: ONLY USING THE TEST DATA
    inptL = inptL[:numbdatatest,:,None]
    inptG = inptG[:numbdatatest,:,None]
    y_pred = model.predict([inptL, inptG])

    y_real = outp[:numbdatatest]

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
                colr = 'm'

            else:
                typstr = 'train'
                colr = 'g'


            print("Epoch: {0}, ".format(epoc) + typstr) 

            for threshold in trange(len(thresh)):
                # this is wrong ***
                x, y = metr[epoc, threshold, i, 2], metr[epoc, threshold, i, 0]

                if not np.isnan(x) and x != 0 and not np.isnan(y) and y != 0:
                    x_points.append(x) # recall
                    y_points.append(y) # precision
                    axis.plot(x, y, marker='o', ls='', markersize=3, alpha=0.6, color=colr)


    
    axis.axhline(1, alpha=.4)
    axis.axvline(1, alpha=.4)
    props = dict(boxstyle='round', alpha=0.4)
    axis.text(0.05, 0.25, textbox, transform=axis.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision v Recall')


    with cd(outpdir):
        with cd(pvrdir):
            plt.savefig('PvR{0}.pdf'.format(path_namer_str))
            plt.close()
    
    return None


# -----------------------------------------------------------------------------------

confdir = 'Confusion_Matrix'

disk_folder(confdir, (outpdir), overwrite=overwrite)

# graphs conf matrix per epoch
# remake to just take conf_matr_vals
def graph_conf():

    with cd(outpdir):
        with cd(matrdir):
            matrconf = np.load(pathsaveconf)


    fig, axis = plt.subplots(2, 2, constrained_layout=True, figsize=(12,6))

    print("Graphing inpt based on conf_matr") 
    
    shape = 'v'

    # constant threshold needed to get just one set of values\ unneeded?
    const_thresh = 0.7

    for epoch in trange(len(matrconf)):

        trne = matrconf[epoch][0]
        flpo = matrconf[epoch][1]
        flne = matrconf[epoch][0]
        trpo = matrconf[epoch][1]

        
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

    with cd(outpdir):
        with cd(confdir):
            plt.savefig('inptspace_confmatr' + inptb4path)
            plt.close()

# -----------------------------------------------------------------------------------
def make_folders():
    # put all the folder stuff here, make global, and move this function to the top
    pass

def main():

    # include logic about seeing if files are loaded, if overwritting needs to happen
    # where to pick up, whatnot
    load_start()
    gen_binned()
    inpt_before_train()
    gen_metr()
    graph_PvR()
    graph_conf()



if __name__ == "__main__":
    main()

    with cd(outpdir):
        with cd(matrdir):
            metr = np.load(pathsavemetr)
            conf = np.load(pathsaveconf)

            print('metr\n', np.shape(np.reshape(metr, (-1,3))), '\nconf\n', np.shape(np.reshape(conf, (-1,4))))