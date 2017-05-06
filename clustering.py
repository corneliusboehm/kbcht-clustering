import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
from scipy.io.arff import loadarff
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import euclidean
from sklearn import metrics
from sklearn.cluster import KMeans
import sys


############################## utility functions ###############################

def load_data(file):
    all_data, meta = loadarff(file)
    
    # split data and class attribute
    data = np.array([e.tolist()[:-1] for e in all_data])
    labels = np.array([int(e.tolist()[-1]) for e in all_data])
    
    return data, labels

def create_clusters(data, assignments):
    return [data[assignments == i] for i in np.unique(assignments)]

def evaluate(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels_pred)

def visualize(data, ax=None, title=''):
    if ax is None:
        # create a new axis if no existing one is provided
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.set_title(title)
    
    if type(data) == np.ndarray:
        # there is only one cluster -> plot directly
        ax.plot(data[:,0], data[:,1], '.')
    else:
        # there are multiple clusters -> plot each one in a different color
        for c in data:
            ax.plot(c[:,0], c[:,1], '.')

def visualize_vertex(vertex, inside, ax=None, title=''):
    if ax is None:
        # create a new axis if no existing one is provided
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.set_title(title)

    # plot hull vertex connected by lines
    ax.plot(vertex[:,0], vertex[:,1], 'x-')

    # plot points inside
    ax.plot(inside[:,0], inside[:,1], '.')


############################### KBCHT algorithm ################################

def kmeans(data, k):
    km = KMeans(n_clusters=k).fit(data)
    return km

def convex_hull(initial_cluster):
    ch = ConvexHull(initial_cluster)
    hull_indices = ch.vertices
    inside_indices = list(set(range(0, len(initial_cluster))) - \
                          set(hull_indices))
    
    initial_vertex = initial_cluster[hull_indices]
    inside = initial_cluster[inside_indices]

    # TODO: always add first node to the end?
    
    return initial_vertex, inside

def average_distance(cluster):
    # TODO: maybe check if the cluster is empty

    # calculate avg edge length in inner points (via delaunay triangulation)
    dt = Delaunay(cluster)

    # get all edges in the triangulation
    edges = set()
    vnv = dt.vertex_neighbor_vertices
    for vi in range(len(vnv[0])-1):
        # for each vertex in the triangulation get its neighbors
        nvis = vnv[1][vnv[0][vi]:vnv[0][vi+1]]
        # create tuples with neighbors
        for nvi in nvis:
            # add edge to set
            edges.add((vi,nvi))

    # compute distances and return average distance
    edge_lengths = [euclidean(dt.points[v[0]], dt.points[v[1]]) for v in edges]
    avg_edge_length = np.mean(edge_lengths)

    return avg_edge_length

def shrink_vertex(initial_vertex, inside):
    # max edge length in convex hull
    edge_lengths = [euclidean(v[0], v[1]) for v in 
        zip(initial_vertex, 
            np.append(initial_vertex[1:], [initial_vertex[0]], axis=0))
    ]
    max_edge_idx = np.argmax(edge_lengths)
    max_edge_length = np.max(edge_lengths)
    #return edge_lengths.index(max_edge_length), max_edge_dist

    avg_edge_length = average_distance(inside)

    if max_edge_length < avg_edge_length:
        # ignore current hull, compute new one from remaining points
        # TODO: what about the previous hull?
        return shrink_vertex(convex_hull(inside))

    # shift convex hull to have the longest edge at the beginning
    # the scipy implementation puts vertices already in counterclockwise order
    initial_vertex = np.roll(initial_vertex, -max_edge_idx, axis=0)

    # shrinking
    V1 = initial_vertex[0]
    V2 = initial_vertex[1]
    ''' WIP
    while max_edge_length >= avg_edge_length or TODO:
        
        candidates = []
        for P in inside:
            # find closest point from x to the line between V1 and V2:
            # 1) its projection falls between V1 and V2
            # 2) it resides on the left of V1 and V2
            # 3) the perpendicular line from P to the line between V1 and V2 doesn't
            # have an intersection with other edgtes between vertices

            # P = V1 + u*(V2-V1)
            V12 = V2-V1
            u = np.dot(P-V1,V12) / np.dot(V12,V12)
            if not (0 <= u <= 1):
                # 1) failed
                continue
            print(u)

        import sys
        sys.exit(0)
    '''

    # for testing only
    shrinked_vertex = initial_vertex
    inside_shrinked = inside

    return shrinked_vertex, inside_shrinked

def points_within(points, vertex):
    if len(points.shape) > 1 and points.shape[1] > 2:
        print('ERROR: Points within can only be found for 2D polygons.')
        return []

    p = Path(vertex)
    return points[p.contains_points(points)]

def find_sub_clusters(shrinked_vertex, inside_shrinked):
    # for testing
    shrinked_vertex = np.append(shrinked_vertex, 
                                [shrinked_vertex[0]], axis=0)

    num_vertices = len(shrinked_vertex)
    cluster_indices = np.zeros(num_vertices)

    cluster_idx = 1

    for i in range(num_vertices-1):
        if cluster_indices[i] == 0:
            for j in range(i+1, num_vertices):
                diff = euclidean(shrinked_vertex[i], shrinked_vertex[j])

                if diff == 0:
                    cluster_indices[range(i, j+1)] = cluster_idx
                    cluster_idx += 1

    # get points inside of subclusters
    sub_clusters = [points_within(inside_shrinked, 
                                  shrinked_vertex[cluster_indices == i]) 
                    for i in range(1, cluster_idx)]
    # TODO: add vertex points to subcluster?

    # calculate average distance for each subcluster
    sc_average_distances = [average_distance(sc) for sc in sub_clusters]
    
    return sub_clusters, cluster_idx-1, sc_average_distances

def parallel_step(initial_cluster):
    initial_vertex, inside = convex_hull(initial_cluster)
    shrinked_vertex, inside_shrinked = shrink_vertex(initial_vertex, inside)
    sub_clusters, sc_length, sc_average_distance = \
        find_sub_clusters(shrinked_vertex, inside_shrinked)
    return sub_clusters, sc_length, sc_average_distance

def get_all_subclusters(initial_clusters):
    sc_tuples = [parallel_step(ic) for ic in initial_clusters]
    sub_clusters = [sub_cluster for t in sc_tuples for sub_cluster in t[0]]
    sc_length = sum([t[1] for t in sc_tuples])
    sc_average_distances = [average_distance for t in sc_tuples 
                                             for average_distance in t[2]]
    
    return sub_clusters, sc_length, sc_average_distances

def merge_clusters(sub_clusters, sc_lengths, sc_average_distances):
    clusters = []
    # TODO: implement
    return clusters

def kbcht(km, data):
    km_clusters = km.predict(data)
    initial_clusters = create_clusters(data, km_clusters)
    sub_clusters, sc_length, sc_average_distances = \
        get_all_subclusters(initial_clusters)
    clusters = merge_clusters(sub_clusters, sc_length, sc_average_distances)
    
    # TODO: This is just for making the process run
    clusters = km_clusters
    
    return clusters


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
    '''
    Usage: python3 clustering.py [K] [FILE]
               K    - Number of clusters for k-means
               FILE - Input file
    '''
    l = len(sys.argv)
    if l == 1:
        k = 2
        file = 'data/c_Moons1.arff'
    elif l == 2:
        try:
            k = int(sys.argv[1])
            file = 'data/c_Moons1.arff'
        except ValueError:
            k = 2
            file = sys.argv[1]
    else:
        try:
            k = int(sys.argv[1])
            file = sys.argv[2]
        except ValueError:
            print('Usage: python3 clustering.py [K] [FILE]\n',
                  '           K    - Number of clusters for k-means\n',
                  '           FILE - Input file')
            sys.exit(0)

    fig = plt.figure(figsize=[10, 4])
    
    print('Load data')
    data, labels_true = load_data(file)
    clusters_true = create_clusters(data, labels_true)
    ax1 = fig.add_subplot(131)
    visualize(clusters_true, ax1, 'True Classes')
    
    print('Perform k-means clustering')
    km = kmeans(data, k)
    labels_pred = km.predict(data)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    clusters_km = create_clusters(data, labels_pred)
    ax2 = fig.add_subplot(132)
    visualize(clusters_km, ax2, 'K-Means Clustering')
    
    print('Perform KBCHT algorithm')
    labels_pred = kbcht(km, data)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    clusters_kbcht = create_clusters(data, labels_pred)
    ax3 = fig.add_subplot(133)
    visualize(clusters_kbcht, ax3, 'KBCHT Clustering')
    
    print('Done')
    # show all plots
    plt.show()
