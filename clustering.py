from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.path import Path
from multiprocessing import Pool
import numpy as np
from scipy.io.arff import loadarff
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist, pdist
from sklearn import metrics
from sklearn.cluster import KMeans
import sys


############################## internal parameter #############################

# tolerance with respect to floating point operations
eps = 0.000001

############################## utility functions ##############################

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
        for idx, cluster in enumerate(clusters):
            if np.any(np.all(cluster == d, axis=1)):
                # this is the first cluster that contains the data point
                assignments.append(idx)
                break

    return assignments

def evaluate(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels_pred)

def visualize(data, ax=None, title='', contains_noise=False):
    if ax is None:
        # create a new axis if no existing one is provided
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.set_title(title)
    
    if type(data) == np.ndarray:
        # there is only one cluster -> plot directly
        if contains_noise:
            ax.plot(data[:,0], data[:,1], 'xk')
        else:
            ax.plot(data[:,0], data[:,1], '.')
    else:
        # there are multiple clusters -> plot each one in a different color
        for c in range(len(data)-1):
            ax.plot(data[c][:,0], data[c][:,1], '.')

        if contains_noise:
            # last cluster is noise
            ax.plot(data[-1][:,0], data[-1][:,1], 'xk')
        else:
            ax.plot(data[-1][:,0], data[-1][:,1], '.')

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

def array_equal(v, w):
    return v[0] == w[0] and v[1] == w[1]

def dist(v, w):
    return sqrt((v[0]-w[0])**2 + (v[1]-w[1])**2)

def seg_intersect(p, r, q, s):
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    rxs = cross2D(r, s)

    if rxs == 0:
        return False

    qp = q - p
    
    t = cross2D(qp, s) / rxs
    if not (0-eps <= t <= 1+eps):
        return False

    u = cross2D(qp, r) / rxs        
    if not (0-eps <= u <= 1+eps):
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
    edge_lengths = [dist(dt.points[v[0]], dt.points[v[1]]) for v in edges]
    avg_edge_length = np.mean(edge_lengths)

    return avg_edge_length

def create_hull(vertices):
    # create hull struct that additionally contains information of edge length
    # (to the next hull vertex) and processed status
    
    dt = np.dtype([('vertex', np.float64, (2,)), 
                   ('length', np.float64), 
                   ('is_processed', bool)])

    hull = np.empty(len(vertices), dtype=dt)
    for i, v in enumerate(vertices):
        j = 0 if i == len(vertices)-1 else i+1
        hull[i] = (v, dist(v, vertices[j]), False)

    return np.rec.array(hull)

def sort_hull(hull):

    max_unprocessed_edge = hull[np.lexsort((-hull.length, hull.is_processed))][0]
    idx = np.where(hull == max_unprocessed_edge)[0][0]

    # shift convex hull to have the longest edge at the beginning
    hull = np.roll(hull, -idx, axis=0)

    return hull, max_unprocessed_edge.length

def shrink_vertex(hull_vertices, inside):

    hull = create_hull(hull_vertices)
    hull, max_edge_length = sort_hull(hull)
    avg_edge_length = average_distance(inside)

    if max_edge_length < avg_edge_length:
        # mark current hull as released, compute new hull from remaining points
        new_hull_vertices, inside = convex_hull(inside)
        new_hull_vertices, released = shrink_vertex(new_hull_vertices, inside)
        return new_hull_vertices, np.append(released, hull_vertices, axis=0)

    all_points = np.append(inside, hull_vertices, axis=0)

    while max_edge_length >= 2*avg_edge_length:
        # shrinking
        V1 = hull[0].vertex
        V2 = hull[1].vertex
        V21 = V2 - V1
        V21dot = np.dot(V21, V21)

        edges = list(
            zip(hull.vertex[1:],
                np.append(hull.vertex[2:], [hull.vertex[0]], axis=0)))

        candidates = []
        for P in all_points:
            # find closest point from x to the line between V1 and V2:
            # 1) its projection falls between V1 and V2
            # 2) it resides on the left of V1 and V2
            # 3) the perpendicular line from P to the line between V1 and V2 doesn't
            # have an intersection with other edges between vertices

            PV1 = P - V1
            u = np.dot(PV1, V21) / V21dot

            if not (0-eps <= u <= 1+eps):
                # 1) failed
                continue

            # NOTE: this only works for 2D
            M = np.vstack((np.array([V1, V2, P]).T,[1,1,1]))
            if np.linalg.det(M) <= 0+eps: # allow some rounding error
                # 2) failed
                continue

            # get projected point
            PP = V1 + u*V21
            PPP = PP - P

            num_intersections = 0
            for i, edge in enumerate(edges):

                if array_equal(P, edge[0]) or array_equal(P, edge[1]):
                    continue

                has_intersection = seg_intersect(P, PPP, edge[0], edge[1]-edge[0])
                if not has_intersection:
                    # no intersection with this edge, therefore check next edge
                    continue

                # we found an intersection. These are only allowed if the
                # candidate vertex is either the V_last or V3...
                if array_equal(P, hull[-1].vertex) or array_equal(P, hull[2].vertex):
                    continue

                # otherwise this is an invalid intersection
                num_intersections += 1
                if num_intersections > 1:
                    # only one intersection can be found at max 
                    # (see condition below)
                    break
            
            if num_intersections == 0 or (num_intersections == 1 and (
                0-eps <= u <= 0+eps or 1-eps <= u <= 1+eps)):
                # Add point if it has no intersection or the only intersection 
                # is at V1 or V2. This happens if u == 0 or u == 1.
                candidates.append((P, dist(P, PP)))            

        if len(candidates) == 0:
            # no candidate for shrinking found
            hull[0].is_processed = True

            if all(hull.is_processed):
                # finished search
                break
        else:
            # add closest point to hull between V1 and V2
            Q = min(candidates, key = lambda t: t[1])[0]
            # update edge length
            hull[0].length = dist(V1, Q)
            hull = np.insert(hull, 1, (Q, dist(Q, V2), False), axis=0)

        hull, max_edge_length = sort_hull(hull)

    return hull.vertex

def release_vertices(vertex):
    released = np.zeros((0, 2))

    change = True
    while change:
        change = False
        to_release = []
        to_remove = []

        num_vertices = len(vertex)
        for i in range(num_vertices):
            if array_equal(vertex[i], vertex[(i+2)%num_vertices]):
                # point at i+1 has same previous and next point 
                # => release the point and remove one of its neighbors
                to_release.append((i+1)%num_vertices)
                to_remove.append(i)
                change = True

        if change:
            to_keep = list(set(range(num_vertices)) 
                           - set(to_release) - set(to_remove))
            
            released = np.append(released, vertex[to_release], axis=0)
            vertex = vertex[to_keep]

    return vertex, released

def points_within(points, vertex):
    # NOTE: this only works for 2D

    if len(points) == 0:
        return np.array([])

    p = Path(vertex)
    within_indices = p.contains_points(points, radius=0.0001)
    return points[within_indices], within_indices

def remove_duplicates(cluster):
    # transform cluster to list of tuples
    l = [tuple(p) for p in cluster]

    # use set to remove duplicates
    s = set(l)

    # transform back to numpy array
    return np.array(list(s))

def find_sub_clusters(shrinked_vertex, initial_cluster):
    # append last vertex point to the end to close the loop
    shrinked_vertex = np.append(shrinked_vertex, 
                                [shrinked_vertex[0]], axis=0)

    num_vertices = len(shrinked_vertex)
    cluster_indices = np.zeros(num_vertices)

    cluster_idx = 1

    for i in range(num_vertices-1):
        last_j = i
        for j in range(i+1, num_vertices):
            if array_equal(shrinked_vertex[i], shrinked_vertex[j]):
                cluster_indices[range(last_j, j+1)] = cluster_idx
                last_j = j
                cluster_idx += 1

    # form subclusters from grouped vertices and points inside them
    sub_clusters = []
    within_indices = np.array([False for _ in initial_cluster])
    for i in range(1, cluster_idx):
        sc_vertices = shrinked_vertex[cluster_indices == i]

        if len(sc_vertices) > 0:
            sc_within, sc_within_indices = \
                points_within(initial_cluster, sc_vertices)
            within_indices = within_indices | sc_within_indices

            if len(sc_within) > 0:
                sc = np.append(sc_vertices, sc_within, axis=0)
            else:
                sc = sc_vertices

            sub_clusters.append(sc)

    # remove duplicates from clusters
    sub_clusters = [remove_duplicates(sc) for sc in sub_clusters]

    # calculate average distance for each subcluster
    sc_average_distances = [average_distance(sc) for sc in sub_clusters]

    # mark points that do not lie within any subcluster as released
    released = initial_cluster[np.where(~within_indices)]
    
    return sub_clusters, sc_average_distances, released

def parallel_step(initial_cluster):
    print('  - Find convex hull')
    initial_vertex, inside = convex_hull(initial_cluster)

    print('  - Shrink vertex')
    shrinked_vertex = shrink_vertex(initial_vertex, inside)
    shrinked_vertex, released = release_vertices(shrinked_vertex)

    print('  - Find subclusters')
    sub_clusters, sc_average_distances, sc_released = \
        find_sub_clusters(shrinked_vertex, initial_cluster)
    released = np.append(released, sc_released, axis=0)

    return sub_clusters, sc_average_distances, released

def get_all_subclusters(initial_clusters):
    sc_tuples = [parallel_step(ic) for ic in initial_clusters]

    # reorganize outputs into separate flattened lists
    sub_clusters = [sub_cluster for t in sc_tuples for sub_cluster in t[0]]
    sc_average_distances = [average_distance for t in sc_tuples 
                                             for average_distance in t[1]]
    released = np.array([r for t in sc_tuples for r in t[2]])
    
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

    if noise is not None:
        clusters.append(noise)

    return clusters, noise is not None

def kbcht(km, data, ax):
    km_clusters = km.predict(data)
    initial_clusters = create_clusters(data, km_clusters)

    print('- Get all subclusters')
    sub_clusters, sc_average_distances, released = \
        get_all_subclusters(initial_clusters)

    visualize(sub_clusters+[released], ax, 'Subclusters', contains_noise=True)

    print('- Merge subclusters')
    clusters, average_distances = \
        merge_clusters(sub_clusters, sc_average_distances)

    clusters, contains_noise = \
        add_released(clusters, average_distances, released)
    
    # recreate cluster assignments for points in original data set
    assignments = create_assignments(data, clusters)
    
    return assignments, contains_noise


############# entry points for the clustering algorithms framework #############

def einfaches_clustering(data, k):
    km = kmeans(data, k)
    list_of_labels, contains_noise = kbcht(km, data, None)
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

    fig = plt.figure(figsize=[12, 4])
    
    print('Load data')
    data, labels_true = load_data(file)
    clusters_true = create_clusters(data, labels_true)
    ax1 = fig.add_subplot(141)
    visualize(clusters_true, ax1, 'True Classes')
    
    print('Perform k-means clustering')
    km = kmeans(data, k)
    labels_pred = km.predict(data)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    clusters_km = create_clusters(data, labels_pred)
    ax2 = fig.add_subplot(142)
    visualize(clusters_km, ax2, 'K-Means Clustering')
    
    print('Perform KBCHT algorithm')
    ax3 = fig.add_subplot(143)
    labels_pred, contains_noise = kbcht(km, data, ax3)
    e = evaluate(labels_true, labels_pred)
    print('Score: {}'.format(e))
    clusters_kbcht = create_clusters(data, labels_pred)
    ax4 = fig.add_subplot(144)
    visualize(clusters_kbcht, ax4, 'KBCHT Clustering', contains_noise)
    
    print('Done')
    # show all plots
    plt.show()
