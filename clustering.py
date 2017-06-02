import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
from scipy.io.arff import loadarff
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import euclidean, cdist, pdist
from sklearn import metrics
from sklearn.cluster import KMeans
import sys


############################## utility functions ###############################

def load_data(file):
    all_data, meta = loadarff(file)
    
    # split data and class attribute, assuming class is the last column
    # TODO: Dynamically check for index of class attribute
    data = np.array([e.tolist()[:-1] for e in all_data])
    labels = np.array([int(e.tolist()[-1]) for e in all_data])

    # only use first two attributes, because KBCHT only works in 2D
    if data.shape[1] > 2:
        print('Only keeping attributes {}, because KBCHT works in 2D'
              .format(meta.names()[:2]))
        data = data[:, :2]
    
    return data, labels

def create_clusters(data, assignments):
    return [data[assignments == i] for i in np.unique(assignments)]

def create_assignments(orig_data, clusters):
    # TODO: there should be a more efficient/pythonic way
    assignments = []

    for d in orig_data:
        for c in range(len(clusters)):
            if np.any(np.all(clusters[c] == d, axis=1)):
                # this is the first cluster that contains the data point
                assignments.append(c)
                break

    return assignments

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

def cross2D(v, w):
    return v[0]*w[1] - v[1]*w[0]

def seg_intersect(p, r, q, s):
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    rxs = cross2D(r, s)

    if rxs == 0:
        return False

    qp = q - p
    
    t = cross2D(qp, s) / rxs
    if not (0 <= t <= 1):
        return False

    u = cross2D(qp, r) / rxs        
    if not (0 <= u <= 1):
        return False

    return True


############################### KBCHT algorithm ################################

def kmeans(data, k):
    km = KMeans(n_clusters=k).fit(data)
    return km

def convex_hull(initial_cluster):
    if len(initial_cluster) < 3:
        return initial_cluster, np.array([])

    ch = ConvexHull(initial_cluster)
    hull_indices = ch.vertices
    inside_indices = list(set(range(len(initial_cluster))) - \
                          set(hull_indices))
    
    initial_vertex = initial_cluster[hull_indices]
    inside = initial_cluster[inside_indices]

    # TODO: always add first node to the end?
    
    return initial_vertex, inside

def average_distance(cluster):
    if len(cluster) <= 1:
        return 0
    if len(cluster) < 4:
        # less than 4 points -> calculate mean distance directly
        return np.mean(pdist(cluster))

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

def sort_hull(vertices):
    # max edge length in convex hull
    edge_lengths = [euclidean(v[0], v[1]) for v in 
        zip(vertices.vertex, 
            np.append(vertices.vertex[1:], [vertices.vertex[0]], axis=0))
    ]

    # indices of by-length sorted array
    by_length = np.argsort(edge_lengths)[::-1]
    max_edge_length = edge_lengths[by_length[0]]
    # sort vertices
    vertices_by_length = vertices[by_length]
    # get longest edge that is not processed
    max_vertex = next(v for v in vertices_by_length if v.is_processed == False)
    max_vertex_idx = np.where(vertices == max_vertex)[0][0]

    # shift convex hull to have the longest edge at the beginning
    vertices = np.roll(vertices, -max_vertex_idx, axis=0)

    return vertices, max_edge_length

def shrink_vertex(hull_vertices, inside):

    # add processed status information
    dt = np.dtype([('vertex', np.float64, (2,)), ('is_processed', bool)])
    hull_vertices = np.rec.array([(v, False) for v in hull_vertices], dtype = dt)

    hull_vertices, max_edge_length = sort_hull(hull_vertices)
    avg_edge_length = average_distance(inside)

    if max_edge_length < avg_edge_length:
        # ignore current hull, compute new one from remaining points
        # TODO: what about the previous hull?
        new_hull, new_inside = convex_hull(inside)
        return shrink_vertex(new_hull, new_inside)

    foo = 0
    while max_edge_length >= avg_edge_length:
        # shrinking
        V1 = hull_vertices[0].vertex
        V2 = hull_vertices[1].vertex
        V21 = V2 - V1
        V21dot = np.dot(V21, V21)

        edges = list(
            zip(hull_vertices.vertex[1:],
                np.append(hull_vertices.vertex[2:], [hull_vertices.vertex[0]], axis=0)))

        candidates = []
        for P in inside:
            # find closest point from x to the line between V1 and V2:
            # 1) its projection falls between V1 and V2
            # 2) it resides on the left of V1 and V2
            # 3) the perpendicular line from P to the line between V1 and V2 doesn't
            # have an intersection with other edges between vertices

            PV1 = P - V1
            u = np.dot(PV1, V21) / V21dot

            if not (0 <= u <= 1):
                # 1) failed
                continue

            # NOTE: this only works for 2D
            M = np.vstack((np.array([V1, V2, P]).T,[1,1,1]))
            if np.linalg.det(M) <= 0.00001: # allow some rounding error
                # 2) failed
                continue

            # get projected point
            PP = V1 + u*V21
            PPP = PP - P

            is_valid = True
            for i, edge in enumerate(edges):

                has_intersection = seg_intersect(P, PPP, edge[0], edge[1]-edge[0])
                if not has_intersection:
                    # no intersection with this edge, therefore check next edge
                    continue

                # we found an intersection. These are only allowed if the
                # candidate vertex is either the V_last or V3...
                _next = 0 if i == len(edges)-1 else i+1
                if np.array_equal(P, hull_vertices[i-1].vertex) or np.array_equal(P, hull_vertices[_next].vertex):
                    continue

                # ... or the intersection is at V1 or V2. This can only happen
                # if the intersection is PP.
                if np.array_equal(PP, V1) or np.array_equal(PP, V2):
                    continue

                # otherwise this is an invalid intersection
                is_valid = False
                break
            
            if is_valid:
                candidates.append((P, np.linalg.norm(PPP)))            

        if len(candidates) == 0:
            # no candidate for shrinking found
            hull_vertices[0].is_processed = True

            if all(hull_vertices.is_processed):
                # finished search
                break
        else:
            # add closest point to hull between V1 and V2
            Q = min(candidates, key = lambda t: t[1])[0]
            hull_vertices = np.insert(hull_vertices, 1, (Q, False), axis=0)

        # TODO release vertices

        hull_vertices, max_edge_length = sort_hull(hull_vertices)

        foo = foo + 1

    # TODO what to do with inside?
    released = []

    return hull_vertices.vertex, inside, released

def points_within(points, vertex):
    # NOTE: this only works for 2D

    if len(points) == 0:
        return np.array([])

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

    # form subclusters from grouped vertices and points inside them
    sub_clusters = []
    for i in range(1, cluster_idx):
        sc_vertices = shrinked_vertex[cluster_indices == i]

        if len(sc_vertices) > 0:
            sc_within = points_within(inside_shrinked, sc_vertices)

            if len(sc_within) > 0:
                sc = np.append(sc_vertices, sc_within, axis=0)
            else:
                sc = sc_vertices

            sub_clusters.append(sc)

    # calculate average distance for each subcluster
    sc_average_distances = [average_distance(sc) for sc in sub_clusters]
    
    return sub_clusters, sc_average_distances

def parallel_step(initial_cluster):
    initial_vertex, inside = convex_hull(initial_cluster)

    shrinked_vertex, inside_shrinked, released = \
        shrink_vertex(initial_vertex, inside)

    sub_clusters, sc_average_distances = \
        find_sub_clusters(shrinked_vertex, inside_shrinked)

    return sub_clusters, sc_average_distances, released

def get_all_subclusters(initial_clusters):
    sc_tuples = [parallel_step(ic) for ic in initial_clusters]
    sub_clusters = [sub_cluster for t in sc_tuples for sub_cluster in t[0]]
    sc_average_distances = [average_distance for t in sc_tuples 
                                             for average_distance in t[1]]
    released = [r for t in sc_tuples for r in t[2]]
    
    return sub_clusters, sc_average_distances, released

def cluster_distance(c1, c2):
    # calculate minimum pairwise distance between the two clusters
    return np.min(cdist(c1, c2))

def merge_clusters(sub_clusters, sc_average_distances):
    nsc = len(sub_clusters)

    if nsc == 0:
        return []

    # calculate minimal distances between all subclusters
    sc_dists = np.zeros([nsc, nsc])
    for i in range(nsc):
        for j in range(i+1, nsc):
            sc_dists[i,j] = cluster_distance(sub_clusters[i], sub_clusters[j])
            sc_dists[j,i] = sc_dists[i,j]

    # create initial cluster from first subcluster
    clusters = [sub_clusters[0]]
    processed_indices = {0}
    average_distances = [sc_average_distances[0]]
    c_dists = np.array([sc_dists[0]])

    # merge clusters until all subclusters have been processed
    while len(processed_indices) < nsc:
        # add subclusters to current cluster as long as they are close enough
        change = True
        while change:
            change = False
            merge_indices, = np.where(c_dists[-1] < average_distances[-1])
            for j in set(merge_indices) - processed_indices:
                change = True
                clusters[-1] = np.append(clusters[-1], sub_clusters[j], axis=0)
                processed_indices.add(j)
            if change:
                average_distances[-1] = average_distance(clusters[-1])
                c_dists[-1] = np.min(sc_dists[merge_indices], axis=0)

        jmin = min(set(range(nsc+1)) - processed_indices)
        if jmin < nsc:
            # create new cluster from the first unprocessed subcluster
            clusters.append(sub_clusters[jmin])
            processed_indices.add(jmin)
            average_distances.append(sc_average_distances[jmin])
            c_dists = np.append(c_dists, [sc_dists[jmin]], axis=0)

    return clusters, average_distances

def add_released(clusters, average_distances, released):
    noise = None
    
    for p in released:
        added = False
        for i, c in enumerate(clusters):
            if cluster_distance(c, [p]) < average_distances[i]:
                clusters[i] = np.append(c, [p], axis=0)
                added = True
                break
        if not added:
            if noise is None:
                noise = np.array([p])
            else:
                noise = np.append(noise, [p], axis=0)

    if not noise is None:
        clusters.append(noise)

    return clusters

def kbcht(km, data):
    km_clusters = km.predict(data)
    initial_clusters = create_clusters(data, km_clusters)

    sub_clusters, sc_average_distances, released = \
        get_all_subclusters(initial_clusters)

    clusters, average_distances = \
        merge_clusters(sub_clusters, sc_average_distances)

    clusters = add_released(clusters, average_distances, released)
    
    # recreate cluster assignments for points in original data set
    assignments = create_assignments(data, clusters)
    
    return assignments


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
