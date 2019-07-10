import os
from time import time
from tdpy.util import summgene
from exop import main as exopmain

import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy.random
import numpy as np

def wrap_clus(funcesti, name):
    
    timeinit = time()
    objt = funcesti.fit(flux)
    #print '%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (name, (time() - t0), estimator.inertia_, \
    #         metrics.homogeneity_score(labels, estimator.labels_), \
    #         metrics.completeness_score(labels, estimator.labels_), \
    #         metrics.v_measure_score(labels, estimator.labels_), \
    #         metrics.adjusted_rand_score(labels, estimator.labels_), \
    #         metrics.adjusted_mutual_info_score(labels,  estimator.labels_), \
    #         metrics.silhouette_score(flux, estimator.labels_, \
    #                                  metric='euclidean', \
    #                                  sample_size=sample_size))
    timefinl = time()
    timediff = timefinl - timeinit
    
    return timediff, objt


pathdata = os.environ['CTHC_DATA_PATH'] + '/unsp/'
os.system('mkdir -p %s' % pathdata)

np.random.seed(0)

# settings
# general
boolmock = False

## PCA
numbcomp = 2
## mock data
numbdata = 1000
stdvflux = 1e-3
dept = 1e-2
numbrele = 200
numbbins = 1001
## clustering
numbclus = 2
liststrgclusruns = ['k-means++', 'random']
## plotting
numbdataplot = 4

# setup
## classes
liststrgclas = ['Ot', 'PC']
numbclas = len(liststrgclas)
indxclas = np.arange(numbclas)

# construct the PCA object
pca = PCA(n_components=numbcomp)

if boolmock:
    # make mock data
    ## phas
    binsphas = np.linspace(0., 1., numbbins + 1)
    meanphas = (binsphas[1:] + binsphas[:-1]) / 2.
    indxdata = np.arange(numbdata)
    indxrele = np.arange(numbrele)
    indxirre = np.arange(numbrele, numbdata)
    flux = np.empty((numbdata, numbbins))
    labltrue = np.zeros(numbdata)
    fact = np.zeros(numbdata)
    s2nr = np.zeros(numbrele)
    deptthis = np.zeros(numbrele)
    for k in indxdata:
        fact[k] = (1. + 2. * np.random.random()) * stdvflux
        flux[k, :] = 1. + fact[k] * np.random.randn(numbbins) 
        if k < numbrele:
            numbtran = np.random.random_integers(15)
            indxtran = np.arange(numbbins / 2 - numbtran / 2, numbbins / 2 + numbtran / 2 + 1)
            deptthis[k] = dept * (1. + np.random.rand())
            flux[k, indxtran] -= deptthis[k]
            s2nr[k] = deptthis[k] / fact[k] * np.sqrt(indxtran.size) 
    labltrue[indxrele] = 1.
    gdat.inptraww, gdat.outp, gdat.peri = exopmain.retr_datamock(numbplan=gdat.numbrele, \
                                                numbnois=gdat.numbirre, numbtime=gdat.numbtime, dept=gdat.dept, nois=gdat.nois, boolflbn=True)
else:
    meanphas, flux, labltrue, legdoutp, tici, itoi = exopmain.retr_datatess(False) 
    indxbadd = np.where(~np.isfinite(flux))[0]
    print 'indxbadd'
    summgene(indxbadd)
    print 'flux'
    summgene(flux)
    flux[indxbadd] = np.random.randn(indxbadd.size)
    
    print 'meanphas'
    summgene(meanphas)
    print 'flux'
    summgene(flux)
    #imp = Imputer(strategy="mean", axis=0)
    #flux = 0wiimp.fit_transformflux))

indxdataplot = np.arange(numbdataplot)

# do clustering
listobjt = []
listlablmodl = []
numbclusruns = len(liststrgclusruns)
indxclusruns = np.arange(numbclusruns)
listtimediff = np.empty(numbclusruns)
listmatrconf = []
for k in indxclusruns:
    timediff, objt = wrap_clus(KMeans(init=liststrgclusruns[k], n_clusters=numbclus, n_init=10), name=liststrgclusruns[k])
    lablmodl = objt.labels_
    matrconf = sklearn.metrics.confusion_matrix(labltrue, lablmodl)
    listtimediff[k] = timediff
    listobjt.append(objt)
    listmatrconf.append(matrconf)
    listlablmodl.append(lablmodl)

print 'flux'
summgene(flux)
print 'np.isfinite(flux).all()'
print np.isfinite(flux).all()
print 'np.isinf(flux).any()'
print np.isinf(flux).any()
print 'np.isnan(flux).any()'
print np.isnan(flux).any()

# do PCA
principalComponents = pca.fit_transform(flux)
for name, valu in pca.__dict__.iteritems():
    print name
    print valu#getattr(pca, name)
    print

# plots
## input
numbframxpos = 3
numbframypos = 7
indxframxpos = np.arange(numbframxpos)
indxframypos = np.arange(numbframypos)
figr, axis = plt.subplots(numbframypos, numbframxpos, figsize=(1.5 * numbframypos, 3 * numbframxpos))

cntr = 0
for i in indxframypos:
    for j in indxframxpos:
        for k in indxdataplot:
            axis[i][j].scatter(meanphas, flux[cntr, :], alpha=0.5, s=1)
            cntr += 1
        if i < 2:
            axis[i][j].set_xticklabels([])
        if j > 0:
            axis[i][j].set_yticklabels([])

axis[-1][numbframxpos/2].set_xlabel('Phase')
axis[numbframypos/2][0].set_ylabel('Relative Flux')
plt.legend()
plt.subplots_adjust(hspace=0, wspace=0)
path = pathdata + 'inpt.png'
print 'Writing to %s...' % path
plt.savefig(path)
plt.close()

# PCA
for k in np.arange(numbclusruns + 1):
    figr, axis = plt.subplots(figsize=(6, 6))
    axis.axvline(0, ls='--', alpha=0.5, color='black')
    axis.axhline(0, ls='--', alpha=0.5, color='black')
    if k == 0:
        indxfrst = indxrele
        indxseco = indxirre
        colrfrst = 'g'
        colrseco = 'purple'
        strgextn = 'true'
        lablextnfrst = 'PC'
        lablextnseco = 'O'
    else:
        indxfrst = np.where(listlablmodl[k-1] == 0)[0]
        indxseco = np.where(listlablmodl[k-1] == 1)[0]
        colrfrst = 'b'
        colrseco = 'r'
        strgextn = 'cla%d' % k
        lablextnfrst = 'PC'
        lablextnseco = 'O'
    axis.scatter(principalComponents[indxfrst, 0], principalComponents[indxfrst, 1], color=colrfrst, s=1, label=lablextnfrst)
    axis.scatter(principalComponents[indxseco, 0], principalComponents[indxseco, 1], color=colrseco, s=1, label=lablextnseco)
    axis.set_xlabel(r'$\theta_1$')
    axis.set_ylabel(r'$\theta_2$')
    plt.legend()
    plt.tight_layout()
    path = pathdata + 'late%s.png' %  strgextn
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()

# confusion matrix
for n in range(2):
    for k in indxclusruns:
        figr, axis = plt.subplots()
        if n == 1:
            temp = listmatrconf[k] / np.sum(listmatrconf[k]).astype(float) * 100.
            strg = 'norm'
        else:
            temp = listmatrconf[k]
            strg = 'raww'
        imag = axis.imshow(temp, interpolation='nearest', origin='lower')
        axis.figure.colorbar(imag, ax=axis)
        axis.set(xticks=indxclas, yticks=indxclas, xticklabels=liststrgclas, yticklabels=liststrgclas, ylabel='True label', xlabel='Predicted label')
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = listmatrconf[k].max() / 2.
        for i in indxclas:
            for j in indxclas:
                axis.text(j, i, temp[i, j], ha="center", va="center", color="white" if listmatrconf[k][i, j] > thresh else "black")
        figr.tight_layout()
        path = pathdata + 'matrconfclu%d%s.png' % (k, strg)
        print 'Writing to %s...' % path
        plt.savefig(path)
        plt.close()

liststrg = ['s2nr', 'fact', 'deptthis']
listlabl = ['SNR', r'$\sigma_w$', '$D$']
for k, valu in enumerate([s2nr, fact, deptthis]):
    figr, axis = plt.subplots()
    imag = axis.hist(valu)
    axis.set_xlabel(listlabl[k])
    figr.tight_layout()
    path = pathdata + 'hist%s.png' % (liststrg[k])
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()


    


