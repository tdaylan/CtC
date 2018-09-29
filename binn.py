import numpy as np

class bind():
    def __init__(self, data, period):
        """ 
        Allows raw data to be passed in, folded over itself, and binned
        """
        self.data = data
        self.period = period

    def binn(self, lowr, high, width):
        """
        Basic creation of bins -- is called within more complex methods later
        """
        bins = []
        for low in range(lowr, high, width):
            bins.append((low, low+width))
        return bins

    def fold(self, divisor):
        """
        Folds the input data in on itself.

        Input:
        divisor := how many data points per period that we enforce

        Output:
        folded := the averaged folded vector over the period
        """

        #initialize
        data = self.data
        period = int(len(data)/divisor)

        # generate holder values
        folded = np.zeros(period)
        noneCtch = np.ones(period) * divisor

        # fill holders with values
        for i in range(len(data)):
            if data[i] == None:
                noneCtch[i%period] -= 1 # catch when a None value occurs -> decreases divisor for averaging later
                pass
            else:
                folded[i%period] += data[i]
        
        # average each element by the number of actual values input
        for i in range(len(folded)):
            folded[i] /= noneCtch[i]

        return folded
    
    def scop(self, binnstep=1, kind='locl', period=10):
        """
        scop = Bin within a certain Scope

        Inputs:
        binnstep := how wide you want the bins to be
        kind := locl or globl (local or global) -> large or small scope
        period := if local, this is the local period

        Outputs:
        binned list as a np.array
        """

        # test kind
        if kind == 'globl':
            divisor = 1
        elif kind == 'locl':
            divisor = int(len(self.data)/period)
        else:
            # don't cause a break if kind is wrong, just correct and notify
            print('This kind is not recognized, using local magnification')
            self.scop()
        
        # call folded to get the right scope
        folded = self.fold(divisor)

        # bounds of the binning
        lowr = 0
        high = len(folded)

        # bin the data
        temp = self.binn(lowr, high, binnstep)

        # holder
        final = []

        # iterate through each bin to average the values in the bin
        for rang in temp:
            summer = 0
            for i in range(rang[0],rang[1]):
                summer += self.data[i]
            final.append(summer/(rang[1]-rang[0]))
        
        return np.asarray(final)