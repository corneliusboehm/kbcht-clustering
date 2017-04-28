import numpy as np
from scipy.io.arff import loadarff
from sklearn import metrics
from sklearn.cluster import KMeans


############################## utility functions ###############################

def load_data(f):
    data, meta = loadarff(f)
    
    # split data and class attribute
    d = np.array([e.tolist()[:-1] for e in data])
    l = np.array([int(e.tolist()[-1]) for e in data])
    
    return d, l

def evaluate(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels_pred)


############################### KBCHT algorithm ################################

def kmeans(d, k):
    km = KMeans(n_clusters=k).fit(d)
    return km


############# entry points for the clustering algorithms framework #############

def einfaches_clustering(data, arg1, arg2):
    # TODO: implement
    list_of_labels = []
    return list_of_labels

def tolles_clustering_mit_visualisierung(data, arg1, arg2):
    # TODO: implement
    list_of_labels = []
    list_of_image_filenames = []
    return list_of_labels, list_of_image_filenames


if __name__ == "__main__":
    print('Load data')
    d, labels_true = load_data('data/1-training.arff')
    
    print('Perform k-means clustering')
    km = kmeans(d, 3)
    labels_pred = km.predict(d)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    
    print('Done')

