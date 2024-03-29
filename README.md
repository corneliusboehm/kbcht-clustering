# K-Means Based Convex Hull Triangulation Clustering Algorithm

Python implementation of the KBCHT algorithm [1]. The general algorithm flow may be summarised as follows:

>In the first phase we use the well-known Kmeans algorithm for its simplicity and speed in practice. The second phase takes the initial groups of first phase for processing them in a parallel fashion using shrinking based  on the convex hull of the initial groups. Hence, the third phase considers the sub-clusters obtained from second phase for merging process based on the Delaunay triangulation.
&mdash; <cite>Abubaker, M. B., & Hamad, H. M. (2012)</cite>[1]

## Usage

The implementation follows the scikit-learn [2] style and can be used equivalently to other algorithms of the `sklearn.cluster` module. See example given below.

The KBCHT algorithm can be also directly executed by calling the `kbcht()` function. See example given below.

In addition to the labels, the implementation also provides visualizations for several stages of the clustering process. See example given below.

## Examples

Using sklearn style
```python
>>> import numpy as np
>>> from kbcht import *
>>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
...               [5, 5], [5, 6], [6, 5], [6, 6]])
>>> kbcht = KBCHT(k=2).fit(X)
>>> kbcht.labels_
array([1, 1, 1, 1, 0, 0, 0, 0])
```
Directly calling function
```python
>>> import numpy as np
>>> from kbcht import *
>>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
...               [5, 5], [5, 6], [6, 5], [6, 6]])
>>> labels, _ = kbcht(X, k=2)
>>> labels
array([1, 1, 1, 1, 0, 0, 0, 0])
```
Visualizations for the _Moons_ data set
```python
>>> from sklearn.datasets import make_moons
>>> from kbcht import *
>>> X, _ = make_moons(3000, True, 0.05, 1)
>>> kbcht = KBCHT(k=10, shrinking_threshold=4).fit(X)
>>> for v in kbcht.visualizations:
...     v.show()
```
<img src="pics/kmeans_clustering.png" width="300"> <img src="pics/shrinked_vertices.png" width="300">
<img src="pics/subclusters.png" width="300"> <img src="pics/kbcht_clustering.png" width="300">

---
[1]: Abubaker, M. B., & Hamad, H. M. (2012). "K-means-based convex hull     triangulation clustering algorithm". Research Notes in Information Science, 9(1), 19-29. http://www.globalcis.org/rnis/ppl/RNIS105PPL.pdf

[2]: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825-2830. http://scikit-learn.org