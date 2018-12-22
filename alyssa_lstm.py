from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import LSTM
from exop.main import retr_datamock
from keras.datasets import imdb
from keras.optimizers import RMSprop
from main import gdatstrt
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import tensorflow as tf

def precision(threshold, preds, classification):
	tp = 0
	fp = 0
	for i in range(len(preds)):
		if preds[i] >= threshold:
			if classification[i] == 1:
				tp += 1
			else:
				fp += 1
	if tp + fp == 0:
		return 0
	return tp/(tp+fp)


def recall(threshold, preds, classification):
	tp = 0
	fn = 0
	for i in range(len(preds)):
		if preds[i] >= threshold:
			if classification[i] == 1:
				tp += 1
		else:
			if classification[i] == 1:
				fn += 1
	if tp + fn == 0:
		return 0
	return tp/(tp+fn)

def model(foldername):
	"""
	foldername: name of folders to save models in
	"""
	numbepocs = 20
	#noises = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
	noises = [1e-6, 1e-3, 1e-1]
	perctplan = [.5]
	numbneig = 4
	for nois in noises:
		nois_auc = []
		for perct in perctplan:
			#print ("HELLO")

			aucs = []
			inpttran, outptran, peri = retr_datamock(numbplan=int(perct*100), numbnois=int((1-perct)*100), nois = nois, lstm = True)
			#print ("DATA")
			updtinpt = []
			updtoutp = []
			inpttest, outptest, peri = retr_datamock(numbplan=5, numbnois=0, nois = nois, lstm = True)
			model = Sequential()
			model.add(LSTM(256))
			model.add(Dense(1, activation = 'sigmoid'))
			model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
			
			for i in range(len(inpttran)):
				currinpt = []
				curroutp = []
				for a in range(numbneig, len	(inpttran[i])-numbneig+1):						
					inpt = inpttran[i][a-numbneig:a+numbneig]
					currinpt.append(inpt)
					if 1 in outptran[i][int(a-numbneig/2):int(a+numbneig/2+1)]:
						curroutp.append([1])
					else:
						curroutp.append([0])
				updtinpt.append(currinpt)
				updtoutp.append(curroutp)
			#print (len(updtoutp))
			#print (updtoutp[0])
			#print (len(updtinpt))
			#print (len(updtinpt[0]))
			#print (updtinpt[0][0])
			model.fit(updtinpt[0], updtoutp[0], epochs = 20, batch_size = 10)
			#print ("HELLO")
			modelname = "models/" + foldername + "/nois_" + str(nois) + "_perct_" + str(perct) 
			#print (modelname)
			model.save(modelname)	

"""
def model1(filename):
	myFile = open(filename, 'w')
	with myFile:
		writer = csv.writer(myFile)
		writer.writerows([['nois', 'ratio', 'precisions', 'recalls', 'auc']])
		numbepocs = 20
		noises = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
		#perctplan = [.1, .3, .5, .7, .9]
		noises = [1e-6, 1e-3, 1e-1]
		perctplan = [.5]
		#perct = .5
		#noises = [1e-6, 1e-3, 1e-1]
		#perctplan = [.3, .5, .7]
		layrsize = []
		auc_results = []
		#max duration of transit is 4
		numbneig = 4
		for nois in noises:
			nois_auc = []
			for perct in perctplan:
				aucs = []
				inpttran, outptran, _ = retr_datamock(numbplan=int(perct*100), numbnois=int((1-perct)*100), nois = nois, lstm = True)
				#inpttran, outptran = retr_datamock(numbplan=1, numbnois=0, nois = nois, lstm = True)
				updtinpt = []
				updtoutp = []
				inpttest, outptest, _ = retr_datamock(numbplan=5, numbnois=0, nois = nois, lstm = True)
				model = Sequential()
				model.add(LSTM(256))
				#model.add(Dense(256, activation = ?)) 
				model.add(Dense(1, activation = 'sigmoid'))
				model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

				for i in range(len(inpttran)):
					currinpt = []
					curroutp = []
					for a in range(numbneig, len	(inpttran[i])-numbneig+1):						
						inpt = inpttran[i][a-numbneig:a+numbneig]
						currinpt.append(inpt)
						if 1 in outptran[i][int(a-numbneig/2):int(a+numbneig/2+1)]:
							curroutp.append([1])
						else:
							curroutp.append([0])
						#update = (np.random.rand(len(outptran[i])) > 0.9).astype(int)
						#print (update)
						#outp = np.array(np.array(update))
						#outp = np.array(np.array(outptran[i]))
						#fig, ax = plt.subplots()
						#print (inpttran[i])
						#print (outptran[i])
					#print (inpttran[i])
					#print (currinpt)
					#print (outptran[i])
					#print (curroutp)
					updtinpt.append(currinpt)
					updtoutp.append(curroutp)

					for a in range(len(inpttran[i])):
						if outptran[i][a] == 0:
							ax.scatter(a, inpttran[i][a], color='red')
						else:
							ax.scatter(a, inpttran[i][a], color='blue')
					name = 'figures/trandata_' + str(i) + "_" + filename + '.png'
					#plt.savefig(name)

				#inpt = np.array(inpttran)
				#outp = np.array(outptran)
				model.fit(updtinpt, updtoutp, epochs = 20, batch_size = 10)


				for i in range(len(inpttest)):
					currinpt= []
					curroutp = []
					for a in range(numbneig, len(inpttest[i])-numbneig+1):						
						inpt = inpttest[i][a-numbneig:a+numbneig]
						currinpt.append(inpt)
						if 1 in outptest[i][int(a-numbneig/2):int(a+numbneig/2+1)]:
							curroutp.append([1])
						else:
							curroutp.append([0])
					scores = model.evaluate(currinpt, curroutp, verbose=0)
					preds = model.predict(currinpt).ravel()
				for i in range(len(inpttest)):
					inpt = inpttest[i][:, None, None]
					b = np.array(outptest[i])
					fig, ax = plt.subplots()
					for a in range(len(inpttest[i])):
						if outptest[i][a] == 0:
							ax.scatter(a, inpttest[i][a], color='red')
						else:
							ax.scatter(a, inpttest[i][a], color='blue')
					name = 'figures/testdata_' + str(nois) + "_" + str(i) + "_" + filename + '.png'
					#plt.savefig(name)
					scores = model.evaluate(inpt, b, verbose=0)
					preds = model.predict(inpt).ravel()
					for e in range(len(inpttest[i])):
						#print ("IN", inpttest[i][e])
						#print ("PRED", preds[e])
						#print ("OUT", outptest[i][e])
						pass
					fig, ax = plt.subplots()
					for a in range(len(inpttest[i])):
						if outptest[i][a] == 0:
							ax.scatter(a, preds[a], color='red')
						else:
							ax.scatter(a, preds[a], color='blue')
					name = 'figures/testresult_' + str(nois) + "_" + str(i) + "_" + filename + '.png'
					#plt.savefig(name)
					fpr_keras, tpr_keras, thresholds_keras = roc_curve(outptest[i], preds)
					auc_keras = auc(fpr_keras, tpr_keras)
					print (auc_keras)
					aucs.append(auc_keras)
					precisions = []
					recalls = []
					for threshold in thresholds_keras:
						recalls.append(recall(threshold, preds, outptest[i]))
						precisions.append(precision(threshold, preds, outptest[i]))
					#print (recalls, precisions)
					writer.writerows([[nois, perct, precisions, recalls, auc_keras]])
					fig, ax = plt.subplots()
					plt.scatter(recalls, precisions)
					ax.set_xlim([0,1])
					ax.set_ylim([0,1])
					ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
					ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
					plt.xlabel("Recalls")
					plt.ylabel("Precisions")
					plt.title('Recall vs Precision')
					plt.tight_layout()
					name = 'figures/' + filename + str(nois) + "_" + str(i) + "_" + str(perct) + '.png'
					#plt.savefig(name)
				nois_auc.append(sum(aucs)/len(aucs))
				print (nois_auc)
				print ("AUC", aucs)
				#print (aucs)
				#print ("AUC AVERAGE", sum(aucs)/len(aucs))
			auc_results.append(nois_auc)
		auc_results.reverse()
		noises.reverse()
		fig, ax = plt.subplots()
		#print (auc_results)
		plt.imshow(auc_results, cmap='hot', interpolation='nearest')
		ax.set_yticks(range(len(noises)))
		ax.set_yticklabels(noises)
		plt.xticks(range(len(perctplan)))
		ax.set_xticklabels(perctplan)
		plt.colorbar()
		plt.ylabel("Nois")
		plt.xlabel("Ratio of Planets to Noise")
		plt.title("AUC heatmap")
		plt.tight_layout()
		name = 'figures/' + filename + 'AUC_heatmap.png'
		plt.savefig(name)
"""
model('12_19_1255')
