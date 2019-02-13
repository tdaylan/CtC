import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras.backend as K
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Reshape, Conv2DTranspose, UpSampling1D, Flatten, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import regularizers

import sys
sys.path.append('/Users/ruginaileana/src')
from exop import main as exopmain
import scipy.stats as ss

from binary_classification_helper import find_km_clusters, plot_confusion_matrix, compute_distance
from plotting_helpers import visualize_1d, visualize_2d, visualize_just_clustering_2d

import time

def model_cnn_autoencoder(ncol, no_filters, kernel_size, pool_size, encoding_dim, activation_function, verbose = False, no_conv = 2, l1_param = 0, l2_param = 0 ):
	"""
	
	"""
	intermediate_size = int(ncol / math.log(ncol))

	#ARCHITECTURE
	input_layer = Input(shape = (ncol,1)); x = input_layer 
	for _ in range(0, no_conv):
		x = Conv1D(no_filters, kernel_size, padding = "same", kernel_regularizer=regularizers.l2(l2_param), activity_regularizer=regularizers.l1(l1_param))(x) 

	x = MaxPooling1D(pool_size = pool_size)(x); _, dim1, dim2 = x.get_shape(); dim1 = int(dim1); dim2 = int(dim2)
	x = Flatten()(x)
	x = Dense(units = encoding_dim, kernel_regularizer=regularizers.l2(l2_param), activity_regularizer=regularizers.l1(l1_param))(x); latent_space = x
	x = Dense(units = dim1 * dim2, kernel_regularizer=regularizers.l2(l2_param), activity_regularizer=regularizers.l1(l1_param))(x)
	x = Reshape((dim1,dim2))(x)

	x = UpSampling1D(size = pool_size)(x)
	for _ in range(0, no_conv):
		x = Conv1D(no_filters, kernel_size, padding = "same", kernel_regularizer=regularizers.l2(l2_param), activity_regularizer=regularizers.l1(l1_param))(x) 
	x = MaxPooling1D(pool_size = no_filters)(x)
	x = Reshape((ncol,1))(x); output_layer = x

	#create both autoencoder and encoder models
	autoencoder = Model(inputs = input_layer, outputs = output_layer )
	encoder = Model(inputs = input_layer, outputs = latent_space )

	if verbose:
		#see what we've done lol
		print ('autoencoder architecture')
		print(autoencoder.summary())

		print ('encoder architecture')
		print(encoder.summary())

	return encoder, autoencoder

def train_cnn_autoencoder(light_curves, autoencoder, patience=50, verbose = False ):
	#split into train and validation
	light_curves_train, light_curves_validation = train_test_split(light_curves, test_size = 0.2)

	#train autoencoder
	early_stopping_monitor = EarlyStopping(patience=patience)
	autoencoder.compile(optimizer = 'adadelta', loss = 'mse')
	history = autoencoder.fit(light_curves_train, light_curves_train, epochs=1000, validation_data=(light_curves_validation, light_curves_validation), 
		callbacks=[early_stopping_monitor], verbose = 0)
	val_loss = history.history['val_loss']
	if verbose:
		plt.plot(range(0, len(val_loss)),  val_loss, linewidth=2)
		plt.title('Validation loss as a function of time')
		plt.show()




def one_datapoint(encoding_dim = 2, no_filters = 4, kernel_size = 3, pool_size = 4, dept=1e-2, nois=1e-3):
	light_curves, labels = exopmain.retr_datamock(numbplan=100, numbnois=100, dept = dept, nois = nois)
	nrow, ncol = light_curves.shape
	intermediate_size = int(ncol / math.log(ncol))

	light_curves = np.reshape(light_curves, (nrow, ncol, 1))
	
	autoencoder_cfms = []
	no_iterations = 5
	for _ in range(0, no_iterations):
		encoder, autoencoder = model_cnn_autoencoder(ncol, no_filters, kernel_size, pool_size, encoding_dim, 'relu', verbose = False)
		train_cnn_autoencoder(light_curves, autoencoder, patience=10, verbose = False)
		latent_repr =  encoder.predict(light_curves)
		clusters = find_km_clusters(latent_repr)
		autoencoder_cfms.append(confusion_matrix(labels, clusters) )

	autoencoder_result = np.mean(autoencoder_cfms, axis = 0)
	autoencoder_std = np.std(autoencoder_cfms, axis = 0)
	
	return autoencoder_result, autoencoder_std / np.sqrt(no_iterations)



def simple_example(clustering = 'km'):
	dept = 100; nois = 1
	light_curves, labels = exopmain.retr_datamock(numbplan=2500, numbnois=2500, dept = dept, nois = nois)
	# plt.plot(range(0, len(light_curves[0])), light_curves[0])
	# plt.show()
	nrow, ncol = light_curves.shape
	print ('number of datapoints: ', nrow)

	intermediate_size = int(ncol / math.log(ncol))
	light_curves = np.reshape(light_curves, (nrow, ncol, 1))

	len_param_range = 1
	l1_param_range = np.logspace(-5, 0, len_param_range)
	l2_param_range = np.logspace(-5, 0, len_param_range)

	matrix = np.zeros((len_param_range, len_param_range))

	for l1_index in range(0, len_param_range):
		for l2_index in range(0, len_param_range):
			l1_param = l1_param_range[l1_index]
			l2_param = l2_param_range[l2_index]

			print ('reg params are: l1 = {}; l2 = {}'.format(l1_param, l2_param))

			encoder, autoencoder = model_cnn_autoencoder(ncol = ncol, no_filters = 4, kernel_size = 3, 
				pool_size = 4, encoding_dim = 2, activation_function = 'relu', verbose = False, l1_param = l1_param, l2_param = l2_param )
			train_cnn_autoencoder(light_curves, autoencoder, patience=10, verbose = False)
			latent_repr =  encoder.predict(light_curves)
		
			if clustering == 'km': clusters = find_km_clusters(latent_repr)
			else: clusters = developing_find_clusters(latent_repr)


			#visualize_2d(latent_repr, labels, clusters, l1_index, l2_index)
			#visualize_just_clustering_2d(latent_repr, labels, l1_index, l2_index)

			plt.imshow(matrix)

def developing_find_clusters(latent_repr):
	"""
	clustering algorithm 
	looks at histogram of distances b/w each point and origin 
	"""
	
	origin = np.zeros(latent_repr[0].shape)
	dist = [compute_distance(origin, point) for point in latent_repr]

	#compute ranks in original distance list, and then sort its entries
	ranks = ss.rankdata(dist)
	sorted_dist = sorted(dist)

	#do the histogram-based clustering
	differences = [sorted_dist[i+1] - sorted_dist[i] for i in range(0, len(sorted_dist) -1)]
	hist, bin_edges = np.histogram(differences)
	print (len(bin_edges))
	cutoff_rank = sum([1 if x > bin_edges[3] else 0 for x in differences]) #so everything in the first bin is non-planet

	#compute and return labels
	labels = [x >= cutoff_rank for x in ranks]
	return labels


simple_example()

# if __name__ == "__main__":
# 	#look at various SNR; default to noise = 1, vary the signal
# 	dept_range = np.geomspace(0.001, 1000000, num = 10)
# 	nois_range = [1 for i in range(0, len(dept_range))]



# 	#default params
# 	no_filters = 2
# 	kernel_size = 20
# 	pool_size = 4


# 	for kernel_size in [10, 20, 30, 40]:
# 		#string with current parameters
# 		# params_str = '\n'.join((
# 	 #    '{} CNN filters'.format(no_filters),
# 	 #    'kernel size is {} '.format(kernel_size) ,
# 	 #    'pool size is {}'.format(pool_size) ))

# 		#print the architecture to file
# 		filename = '___'.join(('{}_CNN_filters'.format(no_filters), 'kernel_size_{} '.format(kernel_size) , 'pool_size_{}'.format(pool_size) ))
# 		encoder, autoencoder = model_cnn_autoencoder(ncol = 100, no_filters = no_filters, kernel_size = kernel_size, pool_size = pool_size, encoding_dim = 2, activation_function = 'relu', verbose = False)

# 		orig_stdout = sys.stdout
# 		f = open(filename, "a")
# 		sys.stdout = f

# 		print ('autoencoder architecture')
# 		autoencoder.summary()
# 		print ('encoder architecture')
# 		encoder.summary()

# 		sys.stdout = orig_stdout
# 		f.close()


# 		#these will hold results
# 		tn_res_arr = [];  fp_res_arr = [];  fn_res_arr = [];  tp_res_arr = []; 
# 		tn_std_arr = [];  fp_std_arr = [];  fn_std_arr = [];  tp_std_arr = []; 

# 		for i in range(0, len(dept_range)):
# 			dept = dept_range[i]; nois = nois_range[i];
# 			autoencoder_result, autoencoder_std = one_datapoint(encoding_dim = 2, no_filters = no_filters, kernel_size = kernel_size, pool_size = pool_size, dept=dept, nois=nois)
# 			print ('done with one datapoint, for one set of parameters')
# 			tn_res, fp_res, fn_res, tp_res = autoencoder_result.ravel()
# 			tn_std, fp_std, fn_std, tp_std = autoencoder_std.ravel()

# 			tn_res_arr.append(tn_res);  fp_res_arr.append(fp_res);  fn_res_arr.append(fn_res);  tp_res_arr.append(tp_res); 
# 			tn_std_arr.append(tn_std);  fp_std_arr.append(fp_std);  fn_std_arr.append(fn_std);  tp_std_arr.append(tp_std); 


# 		plt.figure(figsize=(12, 9))


# 		gs = gridspec.GridSpec(2, 2)

# 		ax1 = plt.subplot(gs[0])
# 		ax1.set_xscale('log')
# 		plt.errorbar(np.divide(dept_range, nois_range), tn_res_arr, yerr=tn_std_arr, fmt='o')
# 		plt.xlabel('SNR')
# 		plt.ylabel('True negative percentage')

# 		ax2 = plt.subplot(gs[1])
# 		ax2.set_xscale('log')
# 		plt.errorbar(np.divide(dept_range, nois_range), fp_res_arr, yerr=fp_std_arr, fmt='o')
# 		plt.xlabel('SNR')
# 		plt.ylabel('False positive percentage')

# 		ax3 = plt.subplot(gs[2])
# 		ax3.set_xscale('log')
# 		plt.errorbar(np.divide(dept_range, nois_range), fn_res_arr, yerr=fn_std_arr, fmt='o')
# 		plt.xlabel('SNR')
# 		plt.ylabel('False negative percentage')

# 		ax4 = plt.subplot(gs[3])
# 		ax4.set_xscale('log')
# 		plt.errorbar(np.divide(dept_range, nois_range), tp_res_arr, yerr=tp_std_arr, fmt='o')
# 		plt.xlabel('SNR')
# 		plt.ylabel('True positive percentage')

# 		plt.savefig('results/' + filename)

# 		print ('done with one set of parameters')





