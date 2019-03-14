import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
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

def model_cnn_autoencoder(ncol, no_filters, kernel_size, pool_size, encoding_dim, activation_function, verbose = False, no_conv = 2, l1_param = 0, l2_param = 0, rel_path = "./"):
	"""
	
	"""

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
		filename = ';'.join((
			'ts_length={}'.format(ncol), 
			'CNN_filters={}'.format(no_filters), 
			'kernel_size={} '.format(kernel_size) , 
			'pool_size={}'.format(pool_size), 
			'encoding_dim = {}'.format(encoding_dim),
			'no_conv={}'.format(no_conv),
			'l1={}'.format(l1_param),
			'l2={}'.format(l2_param)
			))
		orig_stdout = sys.stdout
		f = open(rel_path + filename + '.txt', "a")
		sys.stdout = f

		print ('autoencoder architecture')
		autoencoder.summary()
		print ('encoder architecture')
		encoder.summary()

		sys.stdout = orig_stdout
		f.close()

		return encoder, autoencoder, filename 

	return encoder, autoencoder

def train_cnn_autoencoder(light_curves, autoencoder, patience=50, verbose = False, filename = None, rel_path = "./" ):
	if verbose and filename is None:
		raise ValueError('if verbose is True, need to specify filename')

	#split into train and validation
	light_curves_train, light_curves_validation = train_test_split(light_curves, test_size = 0.2)


	#redirect print statements to file if verbose
	if verbose:
		orig_stdout = sys.stdout
		f = open(rel_path + filename + '.txt', "a")
		sys.stdout = f

	#train autoencoder; 
	early_stopping_monitor = EarlyStopping(patience=patience)
	autoencoder.compile(optimizer = 'adadelta', loss = 'mse')
	history = autoencoder.fit(light_curves_train, light_curves_train, epochs=1000, validation_data=(light_curves_validation, light_curves_validation), 
		callbacks=[early_stopping_monitor], verbose = verbose)
	
	if verbose:
		sys.stdout = orig_stdout
		f.close()

		#also plot validation loss 
		val_loss = history.history['val_loss']
		plt.figure()
		plt.title('Validation Loss')
		plt.plot(range(0, len(val_loss)),  val_loss, linewidth=2)
		plt.savefig(rel_path + filename + '.pdf')

def mock_data_compute_cfms(encoding_dim, no_filters, kernel_size, pool_size, dept, nois, numbtime, no_iterations = 5 ):
	"""
	no_iterations do:
		get mock data from exop; timeseries of length numbtime
		reduce its dimensionality
		apply kmeans 
		look at confusion matrix
	return mean and standard deviation of confusion matrix
	"""
	autoencoder_cfms = []
	for _ in range(0, no_iterations):
		light_curves, labels, _ = exopmain.retr_datamock(numbplan=100, numbnois=100, numbtime = numbtime, dept = dept, nois = nois)
		nrow, ncol = light_curves.shape
		light_curves = np.reshape(light_curves, (nrow, ncol, 1))

		encoder, autoencoder = model_cnn_autoencoder(ncol, no_filters, kernel_size, pool_size, encoding_dim, 'relu')
		train_cnn_autoencoder(light_curves, autoencoder)

		latent_repr =  encoder.predict(light_curves)
		clusters = find_km_clusters(latent_repr)
		autoencoder_cfms.append(confusion_matrix(labels, clusters) )

	autoencoder_result = np.mean(autoencoder_cfms, axis = 0)
	autoencoder_std = np.std(autoencoder_cfms, axis = 0)
	return autoencoder_result, autoencoder_std / np.sqrt(no_iterations)

def mock_data_snr_plots(dept_range, nois_range, encoding_dim = 2, no_filters = 4, kernel_size = 3, pool_size = 4, rel_path = './', numbtime=100):
	"""
	save SNR plot and model architecture in rel_path
	numbtime = length of time series
	"""

	#these will hold results
	tn_res_arr = [];  fp_res_arr = [];  fn_res_arr = [];  tp_res_arr = []; 
	tn_std_arr = [];  fp_std_arr = [];  fn_std_arr = [];  tp_std_arr = []; 

	#print architecture to file
	_, _, filename = model_cnn_autoencoder(ncol = numbtime, no_filters = no_filters, kernel_size = kernel_size, pool_size = pool_size, 
		encoding_dim = encoding_dim, activation_function = 'relu', verbose = True, rel_path = rel_path)

	for i in range(0, len(dept_range)):
		dept = dept_range[i]; nois = nois_range[i];
		autoencoder_result, autoencoder_std = mock_data_compute_cfms(encoding_dim, no_filters, kernel_size, pool_size, dept, nois, numbtime)
		print ('done with one SNR value')
		tn_res, fp_res, fn_res, tp_res = autoencoder_result.ravel()
		tn_std, fp_std, fn_std, tp_std = autoencoder_std.ravel()

		tn_res_arr.append(tn_res);  fp_res_arr.append(fp_res);  fn_res_arr.append(fn_res);  tp_res_arr.append(tp_res); 
		tn_std_arr.append(tn_std);  fp_std_arr.append(fp_std);  fn_std_arr.append(fn_std);  tp_std_arr.append(tp_std); 


	plt.figure(figsize=(12, 9))

	gs = gridspec.GridSpec(2, 2)

	ax1 = plt.subplot(gs[0])
	ax1.set_xscale('log')
	plt.errorbar(np.divide(dept_range, nois_range), tn_res_arr, yerr=tn_std_arr, fmt='o')
	plt.xlabel('SNR')
	plt.ylabel('True negative percentage')

	ax2 = plt.subplot(gs[1])
	ax2.set_xscale('log')
	plt.errorbar(np.divide(dept_range, nois_range), fp_res_arr, yerr=fp_std_arr, fmt='o')
	plt.xlabel('SNR')
	plt.ylabel('False positive percentage')

	ax3 = plt.subplot(gs[2])
	ax3.set_xscale('log')
	plt.errorbar(np.divide(dept_range, nois_range), fn_res_arr, yerr=fn_std_arr, fmt='o')
	plt.xlabel('SNR')
	plt.ylabel('False negative percentage')

	ax4 = plt.subplot(gs[3])
	ax4.set_xscale('log')
	plt.errorbar(np.divide(dept_range, nois_range), tp_res_arr, yerr=tp_std_arr, fmt='o')
	plt.xlabel('SNR')
	plt.ylabel('True positive percentage')

	plt.savefig(rel_path + filename  +'.pdf')


def plot_input_ts(light_curves):
	plt.figure()
	for i in range(0,5):
		plt.plot(range(0, len(light_curves[i])), light_curves[i])
	plt.title('Input Time Series')
	plt.savefig(save_path + 'input.pdf')

if __name__ == "__main__":
	#set parameters
	no_filters = 4; kernel_size = 3; pool_size = 4; encoding_dim = 2
	l1_param = 0.1; l2_param = 0.1
	usetess = True
	
	save_path = 'ileana_output_files/tess_data/'

	if usetess:
		_, light_curves, labels, _, _,_ = exopmain.retr_datatess(True, boolplot = False)
	else:
		dept = 1e-2; nois = 1e-4
		light_curves, labels, _ = exopmain.retr_datamock(numbplan=100, numbnois=100, dept = dept, nois = nois)
	#plot_input_ts(light_curves, save_path)
	nrow, ncol = light_curves.shape
	light_curves = np.reshape(light_curves, (nrow, ncol, 1))

	encoder, autoencoder, filename = model_cnn_autoencoder(ncol = ncol, no_filters = no_filters, kernel_size = kernel_size, 
				pool_size = pool_size, encoding_dim = encoding_dim, activation_function = 'relu', verbose = True, 
				l1_param = l1_param, l2_param = l2_param, rel_path = save_path + 'architecture/' )
	train_cnn_autoencoder(light_curves, autoencoder, verbose = True, filename = filename, rel_path = save_path + 'training/')
	latent_repr =  encoder.predict(light_curves)
   
	colors =  ["red" if x else "blue" for x in labels] #planet transits are red; others are blue
	plt.figure()
	plt.scatter(latent_repr[:, 0],latent_repr[:, 1], color=colors, s = 5)
	red_patch = mpatches.Patch(color="red", label='Signal'); blue_patch = mpatches.Patch(color="blue", label='No Signal')
	plt.legend(handles=[red_patch, blue_patch], loc='upper right')
	plt.savefig(save_path + 'latent_space_repr/' + filename + '.pdf')

