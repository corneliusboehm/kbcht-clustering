import numpy as np
from scipy.io.arff import loadarff
from sklearn.cluster import KMeans


def load_data(f):
    d, meta = loadarff(f)
    
    # remove class attribute from data
    d = np.array([e.tolist()[:-1] for e in d])
    
    return d

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
    d = load_data('data/1-training.arff')
    
    print('Perform k-means clustering')
    km = kmeans(d, 3)
    
    print('Done')

