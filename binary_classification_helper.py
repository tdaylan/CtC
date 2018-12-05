from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import itertools



def compute_distance(p1, p2):
	"""
	computes euclidean distance between two points, p1 and p2
	"""
	diff = p1 - p2
	return np.dot(diff, diff)

def compute_inertia(cluster, center):
	"""
	computes inertia of a given cluster
	"""
	distances = [compute_distance(point, center) for point in cluster]
	return sum(distances)

def find_km_clusters(X):
	km = KMeans(n_clusters=2)
	clusters = km.fit_predict(X)

	center_0 = km.cluster_centers_[0,:]
	center_1 = km.cluster_centers_[1,:]

	no_points = X.shape[0]
	group_0 = [X[i] for i in range(0, no_points) if clusters[i] == 0]
	group_1 = [X[i] for i in range(0, no_points) if clusters[i] == 1]

	inertia0 = compute_inertia(group_0, center_0) / len(group_0)
	inertia1 = compute_inertia(group_1, center_1) / len(group_1)

	if inertia0 <= inertia1:
		return clusters
	else:
		return (clusters + 1) % 2


#taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('plots/' + title)