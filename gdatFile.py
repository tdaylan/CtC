import numpy as np

class gdatstrt(object):
    """
    Initializes all the testing data
    """
    def __init__(self):
        # fraction of data samples that will be used to test the model
        self.fractest = 0.1
    
        # number of epochs
        self.numbepoc = 50
    
        # number of runs for each configuration in order to determine the statistical uncertainty
        self.numbruns = 2

        self.indxepoc = np.arange(self.numbepoc)
        self.indxruns = np.arange(self.numbruns)


        # a dictionary to hold the variable values for which the training will be repeated
        self.listvalu = {}
        # relating to the data
        self.listvalu['numbtime'] = np.array([3e0, 1e1, 3e1, 1e2, 3e2]).astype(int)
        self.listvalu['dept'] = 1 - np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        self.listvalu['nois'] = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        self.listvalu['numbdata'] = np.array([1e2, 3e2, 1e3, 3e3, 1e4]).astype(int)
        self.listvalu['fracplan'] = [0.1, 0.3, 0.5, 0.6, 0.9]
        
        # hyperparameters
        self.listvalu['numbdatabtch'] = [32, 128, 512]
        self.listvalu['numbdimsfrst'] = [32, 64, 128]
        self.listvalu['numbdimsseco'] = [32, 64, 128]
        self.listvalu['fracdropfrst'] = [0.25, 0.5, 0.75]
        self.listvalu['fracdropseco'] = [0.25, 0.5, 0.75]
        
        # list of strings holding the names of the variables
        self.liststrgvarb = self.listvalu.keys()
        
        self.numbvarb = len(self.liststrgvarb) 
        self.indxvarb = np.arange(self.numbvarb)
        
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
        