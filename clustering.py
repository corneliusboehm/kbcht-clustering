import matplotlib.pyplot as plt
import numpy as np
from scipy.io.arff import loadarff
from scipy.spatial import ConvexHull
from sklearn import metrics
from sklearn.cluster import KMeans


############################## utility functions ###############################

def load_data(f):
    data, meta = loadarff(f)
    
    # split data and class attribute
    d = np.array([e.tolist()[:-1] for e in data])
    l = np.array([int(e.tolist()[-1]) for e in data])
    
    return d, l

def create_clusters(d, assignments):
    return [d[assignments == i] for i in np.unique(assignments)]

def evaluate(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels_pred)

def visualize(d, ax=None, title=''):
    if ax is None:
        # create a new axis if no existing one is provided
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.set_title(title)
    
    if type(d) == np.ndarray:
        # there is only one cluster -> plot directly
        ax.plot(d[:,0], d[:,1], '.')
    else:
        # there are multiple clusters -> plot each one in a different color
        for c in d:
            ax.plot(c[:,0], c[:,1], '.')


############################### KBCHT algorithm ################################

def kmeans(d, k):
    km = KMeans(n_clusters=k).fit(d)
    return km

def convex_hull(ic):
    ch = ConvexHull(ic)
    hull_indices = ch.vertices
    inside_indices = list(set(range(0, len(ic))) - set(hull_indices))
    
    iv = ic[hull_indices]
    inside = ic[inside_indices]
    
    return iv, inside

def shrink_vertex(iv, inside):
    sv = []
    # TODO: implement
    return sv

def find_sub_clusters(ic, sv):
    sc = []
    s = len(sc)
    scad = 0
    # TODO: implement
    return sc, s, scad

def parallel_step(ic):
    iv, inside = convex_hull(ic)
    sv = shrink_vertex(iv, inside)
    sc, s, scad = find_sub_clusters(ic, sv)
    return sc, s, scad

def get_all_subclusters(ics):
    ts = [parallel_step(ic) for ic in ics]
    scs = [t[0] for t in ts]
    ss = [t[1] for t in ts]
    scads = [t[2] for t in ts]
    
    return scs, ss, scads

def merge_clusters(scs, ss, scads):
    c = []
    # TODO: implement
    return c

def kbcht(km, d):
    km_clusters = km.predict(d)
    ics = create_clusters(d, km_clusters)
    scs, ss, scads = get_all_subclusters(ics)
    c = merge_clusters(scs, ss, scads)
    
    # TODO: This is just for making the process run
    c = km.predict(d)
    
    return c


############# entry points for the clustering algorithms framework #############

def einfaches_clustering(data, k):
    km = kmeans(data, k)
    list_of_labels = kbcht(km, data)
    return list_of_labels

def tolles_clustering_mit_visualisierung(data, k):
    # TODO: implement
    list_of_labels = []
    list_of_image_filenames = []
    return list_of_labels, list_of_image_filenames


if __name__ == "__main__":
    fig = plt.figure(figsize=[10, 4])
    
    print('Load data')
    d, labels_true = load_data('data/1-training.arff')
    clusters_true = create_clusters(d, labels_true)
    ax1 = fig.add_subplot(131)
    visualize(clusters_true, ax1, 'True Classes')
    
    print('Perform k-means clustering')
    km = kmeans(d, 3)
    labels_pred = km.predict(d)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    clusters_km = create_clusters(d, labels_pred)
    ax2 = fig.add_subplot(132)
    visualize(clusters_km, ax2, 'K-Means Clustering')
    
    print('Perform KBCHT algorithm')
    labels_pred = kbcht(km, d)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    clusters_kbcht = create_clusters(d, labels_pred)
    ax3 = fig.add_subplot(133)
    visualize(clusters_kbcht, ax3, 'KBCHT Clustering')
    
    print('Done')
    # show all plots are
    plt.show()

