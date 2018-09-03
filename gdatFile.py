import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
import tensorflow as tf

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
        self.numbtime = self.listvalu['numbtime']
        self.listvalu['dept'] = 1 - np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        self.dept = self.listvalu['dept']
        self.listvalu['nois'] = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        self.nois = self.listvalu['nois']
        self.listvalu['numbdata'] = np.array([1e2, 3e2, 1e3, 3e3, 1e4]).astype(int)
        self.numbdata = self.listvalu['numbdata']
        self.listvalu['fracplan'] = [0.1, 0.3, 0.5, 0.6, 0.9]
        self.fracplan = self.listvalu['fracplan']
        
        # hyperparameters
        self.listvalu['numbdatabtch'] = [32, 128, 512]
        self.numbdatabtch = self.listvalu['numbdatabtch']
        self.listvalu['numbdimsfrst'] = [32, 64, 128]
        self.numbdimsfrst = self.listvalu['numbdimsfrst']
        self.listvalu['numbdimsseco'] = [32, 64, 128]
        self.numbdimsseco = self.listvalu['numbdimsseco']
        self.listvalu['fracdropfrst'] = [0.25, 0.5, 0.75]
        self.fracdropfrst = self.listvalu['fracdropfrst']
        self.listvalu['fracdropseco'] = [0.25, 0.5, 0.75]
        self.fracdropseco = self.listvalu['fracdropseco']
        
        # list of strings holding the names of the variables
        self.liststrgvarb = self.listvalu.keys()
        
        self.numbvarb = len(self.liststrgvarb) # number of variables
        self.indxvarb = np.arange(self.numbvarb) # array of all indexes to get any variable
        
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
        
        # trying to condense all class things into one __init__ so all methods can just be called here


        # number of test data samples
        self.numbdatatest = int(self.numbdata * self.fractest)
        # number of training data samples
        self.numbdatatran = self.numbdata - self.numbdatatest
        # number of signal data samples
        numbdataplan = int(self.numbdata * self.fracplan)
        
        # generate (background-only) light curves
        # temp -- this currently does not use the repository 'exop'
        inpt = self.nois * np.random.randn(self.numbdata * self.numbtime).reshape((self.numbdata, self.numbtime)) + 1.
        # set the label of all to 0 (background)
        outp = np.zeros((self.numbdata, 1))
        
        # time indices of the transit 
        ## beginning
        indxinit = int(0.45 * self.numbtime)
        ## end
        indxfinl = int(0.55 * self.numbtime)
    
        # lower the relevant time bins by the transit depth
        inpt[:numbdataplan, indxinit:indxfinl] *= self.dept
        # change the labels of these data samples to 1 (signal)
        outp[:numbdataplan, 0] = 1.
        
        # randomize the data set
        indxdata = np.arange(self.numbdata)
        indxrand = np.random.choice(indxdata, size=self.numbdata, replace=False)
        inpt = inpt[indxrand, :]
        outp = outp[indxrand, :]
        self.inpt = inpt
        self.outp = outp

        # divide the data set into training and test data sets
        numbdatatest = int(self.fractest * self.numbdata)
        self.inpttest = inpt[:numbdatatest, :]
        self.outptest = outp[:numbdatatest, :]
        self.inpttran = inpt[numbdatatest:, :]
        self.outptran = outp[numbdatatest:, :]   

        self.modl = Sequential()

    def addFcon(self, drop = True, whchLayr = True):
        """
        Functionally can be added at any point in the model

        drop: True if Dropout is desired in the model
        whchLayr: True unless last layer, at which point this var needs to be set to 'Last'
        """
        if whchLayr:
            # check to see if this is the last layer, if not, see how many layers precede this next layer
            whchLayr = len(self.modl.layers)

        if whchLayr == 0:
            # if first layer:
            self.modl.add(Dense(self.numbdimsfrst, input_dim=self.numbtime, activation='relu'))
            
            if drop:
                self.modl.add(Dropout(self.fracdropfrst))

        elif whchLayr > 0:
            self.modl.add(Dense(self.numbdimsseco, activation= 'relu'))
            
            if drop:
                self.modl.add(Dropout(self.fracdropseco))

        elif whchLayr == 'Last':
            self.modl.add(Dense(1, activation='sigmoid'))

    def addConv_1D(self, drop = True, whchLayr = True):
        """
        Functionally can be added at any point in the model

        drop: True if Dropout is desired in the model
        whchLayr: True unless last layer, at which point this var needs to be set to 'Last'
        This should not be the last layer!
        """
        if whchLayr:
            # check to see if this is the last layer, if not, see how many layers precede this next layer
            whchLayr = len(self.modl.layers)
        if whchLayr == 0:
            # if first layer:
            self.modl.add(Conv1D(self.numbdimsfrst, kernel_size=self.numbtime, activation='relu')) # should look into these inputs
            
            if drop:
                self.modl.add(Dropout(self.fracdropfrst))

        elif whchLayr > 0:
            self.modl.add(Conv1D(self.numbdimsseco, kernel_size=self.numbtime activation= 'relu')) # should look into these inputs
            
            if drop:
                self.modl.add(Dropout(self.fracdropseco))

        return None

    def get_metr(self, indxvaluthis=None, strgvarbthis=None):     
        """
        Performance method
        """

        self.modl.compile()

        # empt dict
        listvalutemp = {}
        # store with the vars we iterate over
        for o, strgvarb in enumerate(self.liststrgvarb):
            listvalutemp[strgvarb] = self.listvalu[strgvarb][self.numbvalu[o]/2]        
        
        # catch that input and set another val in the dict
        if strgvarbthis != None:
            listvalutemp[strgvarbthis] = self.listvalu[strgvarbthis][indxvaluthis]   

        metr = np.zeros((self.numbepoc, 2, 3)) - 1
        loss = np.empty(self.numbepoc)
        numbepocchec = 5 # hard coded
        

        for y in self.inxepoc:
            hist = self.modl.fit(self.inpt, self.outp, epochs=1, batch_size=self.numbdatabtch, validation_split=self.fractest, verbose=0)
            loss[y] = hist.history['loss'][0]
            indxepocloww = max(0, y- numbepocchec)
            if y == self.gdat.numbepoc - 1 and 100. * (loss[indxepocloww] - loss[y]):
                print('Warning! The optimizer may not have converged.')
                print('loss[indxepocloww]\n', loss[indxepocloww], '\nloss[y]\n', loss[y], '\nloss\n', loss)

            for r in self.gdat.inxrtyp:
                if r==0:
                    inpt = self.inpttran
                    outp = self.outptran
                    numdatatemp = self.numbdatatran
                else:
                    inpt = self.inpttest
                    outp = self.outptest
                    numbdatatemp = self.numbdatatest

                outppred = (self.modl.predict(inpt) > 0.5).astype(int)
                score = self.modl.evaluate(inpt,outp, verpose=0)
                matrconf = confusion_matrix(outp[:, 0], outppred[:, 0])



                trne = matrconf[0, 0]
                flpo = matrconf[0, 1]
                flne = matrconf[1, 0]
                trpo = matrconf[1, 1]


                if float(trpo + flpo) > 0:
                    metr[y, r, 0] = trpo / float(trpo + flpo)
                metr[y, r, 1] = float(trpo + trne) / (trpo + flpo + trne + flne)
                if float(trpo + flne) > 0:
                    metr[y, r, 2] = trpo / float(trpo + flne)
            return metr