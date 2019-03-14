import numpy as np
import matplotlib.pyplot as plt


def visualize_2d(latent_repr, labels, clusters, index_l1, index_l2, rel_path = "./"):
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
	f.suptitle('Conv. Autoencoder')

	colors =  ["red" if x else "blue" for x in labels] #planet transits are red; others are blue
	ax1.scatter(latent_repr[:, 0],latent_repr[:, 1], color=colors, s = 5)
	ax1.set_title('ground truth labels')


	colors =  ["red" if x else "blue" for x in clusters] #planet transits are red; others are blue
	ax2.scatter(latent_repr[:, 0],latent_repr[:, 1], color=colors, s = 5)
	ax2.set_title('clustering results')

	plt.savefig(rel_path + 'regularizers_l1_{}_l2_{}'.format(index_l1, index_l2))

def visualize_1d(latent_repr, labels, clusters,  index_l1, index_l2, rel_path = "./"):
	latent_repr = latent_repr[:,0]
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
	f.suptitle('Conv. Autoencoder')

	red_latent_repr = []
	blue_latent_repr = []
	for index in range(0, len(labels)):
		boolean = labels[index]
		if boolean:
			red_latent_repr.append(latent_repr[index])
		else:
			blue_latent_repr.append(latent_repr[index])
	ax1.hist(red_latent_repr, color="red", alpha = 0.5)
	ax1.hist(blue_latent_repr, color="blue", alpha = 0.5)
	ax1.set_title('ground truth labels')


	red_latent_repr = []
	blue_latent_repr = []
	for index in range(0, len(clusters)):
		boolean = clusters[index]
		if boolean:
			red_latent_repr.append(latent_repr[index])
		else:
			blue_latent_repr.append(latent_repr[index])
	ax2.hist(red_latent_repr, color="red", alpha = 0.5)
	ax2.hist(blue_latent_repr, color="blue", alpha = 0.5)
	ax2.set_title('clustering results')


	plt.savefig(rel_path + 'regularizers_l1_{}_l2_{}'.format(index_l1, index_l2))


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if rho < 0:
    	raise ValueError('rho < 0')
    return (rho, phi)


def visualize_just_clustering_2d(latent_repr, labels, index_l1, index_l2, rel_path = "./"):
	latent_repr_polar = np.array([cart2pol(x[0], x[1]) for x in latent_repr])
	colors =  ["red" if x else "blue" for x in labels] #planet transits are red; others are blue

	red_latent_repr_polar = []
	blue_latent_repr_polar = []
	for index in range(0, len(labels)):
		boolean = labels[index]
		if boolean:
			red_latent_repr_polar.append(latent_repr_polar[index])
		else:
			blue_latent_repr_polar.append(latent_repr_polar[index])

	red_latent_repr_polar = np.array(red_latent_repr_polar)
	blue_latent_repr_polar = np.array(blue_latent_repr_polar)

	plt.figure()
	plt.scatter(latent_repr[:, 0],latent_repr[:, 1], color=colors, s = 5)
	plt.xlabel('latent space unit 1')
	plt.ylabel('latent space unit 2')
	plt.savefig(rel_path + 'cartesian_scatterplots/l1_{}_l2_{}'.format(index_l1, index_l2))

	plt.figure()
	plt.scatter(latent_repr_polar[:, 0],latent_repr_polar[:, 1], color=colors, s = 5)
	plt.xlabel('r')
	plt.ylabel(r'$\theta$')
	plt.savefig(rel_path + 'polar_scatterplots/l1_{}_l2_{}'.format(index_l1, index_l2))

	plt.figure()
	plt.hist(red_latent_repr_polar[:, 0], color="red", alpha = 0.5)
	plt.hist(blue_latent_repr_polar[:, 0], color="blue", alpha = 0.5)
	plt.title('r')
	plt.savefig(rel_path + 'radius_hist/l1_{}_l2_{}'.format(index_l1, index_l2))

	plt.figure()
	plt.hist(red_latent_repr_polar[:, 1], color="red", alpha = 0.5)
	plt.hist(blue_latent_repr_polar[:, 1], color="blue", alpha = 0.5)
	plt.title(r'$\theta$')
	plt.savefig(rel_path + 'angle_hist/l1_{}_l2_{}'.format(index_l1, index_l2))

	



