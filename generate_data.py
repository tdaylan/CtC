import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os 
from astropy.io import fits

def retr_datamock(numbplan=100, numbnois=100, numbtime=100, pathplot=None, booltest=False):
    
    '''
    Function to generate mock light curves.
    numbplan: number of data samples containing signal, i.e., planet transits
    numbnois: number of background data samples, i.e., without planet transits
    numbtime: number of time bins
    pathplot: path in which plots are to be generated
    booltest: Boolean indicating whether the test data is to be returned separately
    '''
    
    # total number of data samples
    numbdata = numbplan + numbnois
    
    indxdata = np.arange(numbdata)
    indxtime = np.arange(numbtime)
    
    # input data
    inpt = np.ones((numbdata, numbtime))
    
    # output data
    outp = np.zeros(numbdata)
   
    # planet transit properties
    ## depths
    dept = 1. - 1e-2 * np.random.random(numbplan)
    ## durations
    dura = np.random.random_integers(1, 4, size=numbplan)
    ## phases
    phas = np.random.random_integers(0, numbtime, size=numbplan)
    ## periods
    peri = np.random.random_integers(6, numbtime, size=numbplan)
    
    # input signal data
    inptplan = np.ones((numbplan, numbtime))
    indxplan = np.arange(numbplan)
    for k in indxplan:
        for t in indxtime:
            if (t - phas[k]) % peri[k] < dura[k]:
                inptplan[k, t] *= dept[k]
    
    # place the signal data
    inpt[:numbplan, :] = inptplan
    outp[:numbplan] = 1.
    
    # add noise to all data
    inpt += 1e-3 * np.random.randn(numbtime * numbdata).reshape((numbdata, numbtime))
    
    if pathplot != None:
        # generate plots if pathplot is set
        print ('Plotting the data set...')
        for k in indxdata:
            figr, axis = plt.subplots()
            axis.plot(indxtime, inpt[k, :])
            axis.set_title(outp[k])
            if k < numbplan:
                for t in indxtime:
                    if (t - phas[k]) % peri[k] == 0:
                        axis.axvline(t, ls='--', alpha=0.3, color='grey')
            axis.set_xlabel('$t$')
            axis.set_ylabel('$f(t)$')
            plt.savefig(pathplot + 'data_%04d.pdf' % k)
            plt.close()
    
    # shuffle the data set
    indxrand = np.random.choice(indxdata, size=numbdata, replace=False) 
    inpt = inpt[indxrand, :]
    outp = outp[indxrand]
   
    if booltest:
        # number of training data samples
        numbdatatran = int(0.8 * numbdata)
        
        # number of test data samples
        numbdatatest = numbdata - numbdatatran
        
        # separate the data set
        inpttran = inpt[:numbdatatran, :]
        outptran = outp[:numbdatatran]
    
        inpttest = inpt[numbdatatran:, :]
        outptest = outp[numbdatatran:]
    
        return inpttran, outptran, inpttest, outptest
    
    else:
        return inpt, outp


def ete6_planet_data():
    """
    param: None
    return:
        dictionary d, summary of all ete6 planet data
        maps tic_ids to number of planets
    """
    #get text data
    planet_data_url = "https://archive.stsci.edu/missions/tess/ete-6/ground_truth/ete6_planet_data.txt"
    data = (urllib.request.urlopen(planet_data_url))
    #create list of lines; decode to ascii
    lines = []
    for line in data:
        lines.append(line.decode('ascii'))
    #create dictionary d
    d = dict()
    #parse each line; ignore comments
    for line in lines:
        if line[0] == "#": 
            pass
        else:
            entries = line.split() #splits by any whitespace char
            tic_id = entries[0]; planet_number = entries[1];

            #pad tic_id
            padded_tic_id = ""
            for i in range(0, 16 - len(tic_id)):
                padded_tic_id += "0"
            padded_tic_id += tic_id

            d[padded_tic_id] = planet_number

    return d
    
    
def get_lightcurve_filenames():
    """
    param: None
    return: 
        dictionary d
        maps tic_id to name of corresponding file 
    """
    d = dict()
    filenames = os.listdir('./data')
    for filename in filenames:
        name, extension = filename.split(".")
        if extension != "fits":
            pass
        else:
            _, tic_id, _, _ = name.split("-")
            d[tic_id] = filename
    return d

def create_dataset():
    planet_data = ete6_planet_data()
    lightcurve_files = get_lightcurve_filenames()   

if __name__ == "__main__":
    create_dataset()