import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from autoencoder import get_latent_vars
sys.path.append('/Users/ruginaileana/src')
from exop import main as exopmain
from binary_classification_helper import find_km_clusters, plot_confusion_matrix

visualize = True
run_all = True
dimensionality_reduction = "PCA" #can also be "PCA" or autoencoder"
lower_dimensionality = 1 #only applies if PCA, for autoencoder always do 2 latent variables


#get data
light_curves, labels = exopmain.retr_datamock()
light_curves = np.array(light_curves)



########################################################################
########################################################################
################PLOTS TO SEE WHAT PCA AND AUTOENCODER DO################
########################################################################
########################################################################

# if run_all:
# 	dimensionality_reduction = "PCA"
# 	lower_dimensionality = 1 
# if dimensionality_reduction == "PCA" and lower_dimensionality == 1:
# 	pca = PCA(n_components=lower_dimensionality)
# 	proj = pca.fit_transform(light_curves)
# 	clusters = find_km_clusters(proj)


# 	if visualize:
# 		f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
# 		percentage = float(pca.explained_variance_ratio_[0])
# 		f.suptitle('First PC, which explains {:.2f} of the variance'.format(percentage))

# 		colors =  ["red" if x else "blue" for x in labels] #planet transits are red; others are blue
# 		ax1.scatter(proj[:, 0], [0 for i in range(0, proj.shape[0])], color=colors, s = 5)
# 		ax1.set_title('ground truth labels')


# 		colors =  ["red" if x else "blue" for x in clusters] #planet transits are red; others are blue
# 		ax2.scatter(proj[:, 0], [0 for i in range(0, proj.shape[0])], color=colors, s = 5)
# 		ax2.set_title('clustering results')

# 		#plt.show()
# 		plt.savefig('plots/kmeans_pca_1d')


# if run_all:
# 	dimensionality_reduction = "PCA" 
# 	lower_dimensionality = 2 
# if dimensionality_reduction == "PCA" and lower_dimensionality == 2:
# 	pca = PCA(n_components=lower_dimensionality)
# 	proj = pca.fit_transform(light_curves)
# 	clusters = find_km_clusters(proj)

# 	if visualize:
# 		f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
# 		p1, p2 = float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])
# 		f.suptitle('First 2PCs, which explain {:.2f} and {:.2f} of the variance'.format(p1, p2))

# 		colors =  ["red" if x else "blue" for x in labels] #planet transits are red; others are blue
# 		ax1.scatter(proj[:, 0], proj[:, 1], color=colors, s = 5)
# 		ax1.set_title('ground truth labels')


# 		colors =  ["red" if x else "blue" for x in clusters] #planet transits are red; others are blue
# 		ax2.scatter(proj[:, 0], proj[:, 1], color=colors, s = 5)
# 		ax2.set_title('clustering results')

# 		#plt.show()
# 		plt.savefig('plots/kmeans_pca_2d')


# if run_all:
# 	dimensionality_reduction = "autoencoder" 
# 	lower_dimensionality = 2 
# if dimensionality_reduction == "autoencoder":
# 	latent_repr = get_latent_vars(light_curves, encoding_dim = 2, extra_layers = 1, activation_function ='linear' , verbose = False)
# 	clusters = find_km_clusters(latent_repr)

# 	if visualize:
# 		f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
# 		f.suptitle('Autoencoder')

# 		colors =  ["red" if x else "blue" for x in labels] #planet transits are red; others are blue
# 		ax1.scatter(latent_repr[:, 0],latent_repr[:, 1], color=colors, s = 5)
# 		ax1.set_title('ground truth labels')


# 		colors =  ["red" if x else "blue" for x in clusters] #planet transits are red; others are blue
# 		ax2.scatter(latent_repr[:, 0],latent_repr[:, 1], color=colors, s = 5)
# 		ax2.set_title('clustering results')

# 		#plt.show()
# 		plt.savefig('plots/kmeans_autoencoder_2d')



########################################################################
########################################################################
#########RUN EACH A BUNCH OF TIMES AND LOOK AT CONFUSION MATRIX#########
########################################################################
########################################################################

no_iterations = 100
pca_1_comp_cfms = []
pca_2_comp_cfms = []
autoencoder_cfms = []
for _ in range(0, no_iterations):
	pca = PCA(n_components=1)
	proj = pca.fit_transform(light_curves)
	clusters = find_km_clusters(proj)
	pca_1_comp_cfms.append( confusion_matrix(labels, clusters) )

	pca = PCA(n_components=2)
	proj = pca.fit_transform(light_curves)
	clusters = find_km_clusters(proj)
	pca_2_comp_cfms.append( confusion_matrix(labels, clusters) )

	latent_repr = get_latent_vars(light_curves, encoding_dim = 2, extra_layers = 1, activation_function ='linear' , verbose = False)
	clusters = find_km_clusters(latent_repr)
	autoencoder_cfms.append(confusion_matrix(labels, clusters) )

pca_1_comp_cfms = np.array(pca_1_comp_cfms);   assert pca_1_comp_cfms.shape == (no_iterations,2,2)
pca_2_comp_cfms = np.array(pca_2_comp_cfms);   assert pca_2_comp_cfms.shape == (no_iterations,2,2)
autoencoder_cfms = np.array(autoencoder_cfms); assert autoencoder_cfms.shape == (no_iterations,2,2)

pca_1_comp_result = np.mean(pca_1_comp_cfms, axis = 0)
pca_2_comp_result = np.mean(pca_2_comp_cfms, axis = 0)
autoencoder_result = np.mean(autoencoder_cfms, axis = 0)

classes = ["negative", "positive"]

# print ('pca 1 component confusion matrix:\n', pca_1_comp_result)
# tn, fp, fn, tp = pca_1_comp_result.ravel()
# print ('true negative is: ', tn, '\n false positive is: ', fp, '\n false negative: ', fn, '\n true positive: ', tp)
# plot_confusion_matrix(pca_1_comp_result, classes, normalize=True, title='confusion_matrix_kmeans_pca_1d', cmap=plt.cm.Blues)

# print ('pca 1 component confusion matrix:\n', pca_2_comp_result)
# tn, fp, fn, tp = pca_2_comp_result.ravel()
# print ('true negative is: ', tn, '\n false positive is: ', fp, '\n false negative: ', fn, '\n true positive: ', tp)
# plot_confusion_matrix(pca_2_comp_result, classes, normalize=True, title='confusion_matrix_kmeans_pca_2d', cmap=plt.cm.Blues)

print ('pca 1 component confusion matrix:\n', autoencoder_result)
tn, fp, fn, tp = autoencoder_result.ravel()
print ('true negative is: ', tn, '\n false positive is: ', fp, '\n false negative: ', fn, '\n true positive: ', tp)
plot_confusion_matrix(autoencoder_result, classes, normalize=True, title='confusion_matrix_kmeans_autoencoder', cmap=plt.cm.Blues)

