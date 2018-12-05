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

import sys
sys.path.append('/Users/ruginaileana/src')
from exop import main as exopmain
from binary_classification_helper import find_km_clusters, plot_confusion_matrix

import time


def model_cnn_autoencoder(ncol, no_filters, kernel_size, pool_size, encoding_dim, activation_function, verbose = False, no_conv = 2):
	"""
	
	"""
	intermediate_size = int(ncol / math.log(ncol))

	#ARCHITECTURE
	input_layer = Input(shape = (ncol,1)); x = input_layer 

	for _ in range(0, no_conv):
		x = Conv1D(no_filters, kernel_size, padding = "same")(x) 

	x = MaxPooling1D(pool_size = pool_size)(x); _, dim1, dim2 = x.get_shape(); dim1 = int(dim1); dim2 = int(dim2)

	x = Flatten()(x)
	x = Dense(units = encoding_dim)(x); latent_space = x
	x = Dense(units = dim1 * dim2)(x)
	x = Reshape((dim1,dim2))(x)

	x = UpSampling1D(size = pool_size)(x)
	for _ in range(0, no_conv):
		x = Conv1D(no_filters, kernel_size, padding = "same")(x) 
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

def train_cnn_autoencoder(light_curves, autoencoder, patience=10, verbose = False ):
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


def visualize():
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
	f.suptitle('Conv. Autoencoder')

	colors =  ["red" if x else "blue" for x in labels] #planet transits are red; others are blue
	ax1.scatter(latent_repr[:, 0],latent_repr[:, 1], color=colors, s = 5)
	ax1.set_title('ground truth labels')


	colors =  ["red" if x else "blue" for x in clusters] #planet transits are red; others are blue
	ax2.scatter(latent_repr[:, 0],latent_repr[:, 1], color=colors, s = 5)
	ax2.set_title('clustering results')

	plt.show()



def one_datapoint(encoding_dim = 2, no_filters = 4, kernel_size = 10, pool_size = 4, dept=1e-2, nois=1e-3):
	light_curves, labels = exopmain.retr_datamock(dept = dept, nois = nois)
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

if __name__ == "__main__":
	dept_range = np.geomspace(0.1, 1000, num=1)
	nois_range = [1e-3 for i in range(0, len(dept_range))]

	#print the architecture once
	model_cnn_autoencoder(ncol = 100, no_filters = 4, kernel_size = 20, pool_size = 4, encoding_dim = 2, activation_function = 'relu', verbose = True)

	tn_res_arr = [];  fp_res_arr = [];  fn_res_arr = [];  tp_res_arr = []; 
	tn_std_arr = [];  fp_std_arr = [];  fn_std_arr = [];  tp_std_arr = []; 

	for dept in dept_range:
		autoencoder_result, autoencoder_std = one_datapoint(encoding_dim = 2, no_filters = 4, kernel_size = 20, pool_size = 4, dept=dept, nois=1e-3)
		print ('done with one datapoint!')
		tn_res, fp_res, fn_res, tp_res = autoencoder_result.ravel()
		tn_std, fp_std, fn_std, tp_std = autoencoder_std.ravel()

		tn_res_arr.append(tn_res);  fp_res_arr.append(fp_res);  fn_res_arr.append(fn_res);  tp_res_arr.append(tp_res); 
		tn_std_arr.append(tn_std);  fp_std_arr.append(fp_std);  fn_std_arr.append(fn_std);  tp_std_arr.append(tp_std); 


	plt.figure(figsize=(12, 9))

	params_str = '\n'.join((
    '{} CNN filters'.format(4),
    'kernel size is {} '.format(20) ,
    'pool size is {}'.format(4) ))

	plt.title('params_str')


	gs = gridspec.GridSpec(2, 2)

	ax1 = plt.subplot(gs[0])
	plt.errorbar(np.divide(dept_range, nois_range), tn_res_arr, yerr=tn_std_arr, fmt='o')
	plt.xlabel('SNR')
	plt.ylabel('False negative percentage')

	ax2 = plt.subplot(gs[1])
	plt.errorbar(np.divide(dept_range, nois_range), fp_res_arr, yerr=fp_std_arr, fmt='o')
	plt.xlabel('SNR')
	plt.ylabel('False positive percentage')

	ax3 = plt.subplot(gs[2])
	plt.errorbar(np.divide(dept_range, nois_range), fn_res_arr, yerr=fn_std_arr, fmt='o')
	plt.xlabel('SNR')
	plt.ylabel('False negative percentage')

	ax4 = plt.subplot(gs[3])
	plt.errorbar(np.divide(dept_range, nois_range), tp_res_arr, yerr=tp_std_arr, fmt='o')
	plt.xlabel('SNR')
	plt.ylabel('True positive percentage')

	

	plt.savefig('plots')





