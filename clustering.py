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

def convex_hull(ic):
    iv = []
    # TODO: implement
    return iv

def shrink_vertex(ic, iv):
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
    iv = convex_hull(ic)
    sv = shrink_vertex(ic, iv)
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
    ics = [d[km_clusters == i] for i in np.unique(km_clusters)]
    
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
    print('Load data')
    d, labels_true = load_data('data/1-training.arff')
    
    print('Perform k-means clustering')
    km = kmeans(d, 3)
    labels_pred = km.predict(d)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    
    print('Perform KBCHT algorithm')
    labels_pred = kbcht(km, d)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    
    print('Done')

