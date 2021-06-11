from numbers import Real
from types import FunctionType

import numpy as np #check all np.array!!
from joblib import Parallel, delayed
import warnings

from gtda.homology import VietorisRipsPersistence
from gtda.utils.intervals import Interval
from gtda.utils.validation import validate_params
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


from gtda.plotting import plot_diagram
from gtda.base import PlotterMixin
from gtda.utils._docs import adapt_fit_transform_docs


@adapt_fit_transform_docs
class LocalVietorisRipsPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """

    Given a :ref:`point cloud <finite_metric_spaces_and_point_clouds>` in
    Eclidean space, or an abstract :ref:`metric space
    <finite_metric_spaces_and_point_clouds>` encoded by a distance matrix,
    information about the local topology around each point is summarized
    in the corresponding dimension vector. This is done by first isolating
    appropriate neighborhoods around each point, then 'coning' off an annulus
    around each point, computing correponding associated persistence dagram,
    and finaly vectorizing the diagrams.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, input data is to be interpreted as a
        distance matrix or adjacency matrix of weighted undirected graphs.
        Otherwise, input data is to be interpreted as a
        a point cloud (i.e. feature arrays), and `metric`
        determines a rule with which to calculate distances between pairs of
        points (i.e. row vectors). If `metric` is a string, it must be one of
        the options allowed by :func:`scipy.spatial.distance.pdist` for its
        metric parameter, or a metric listed in
        :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including
        ``'euclidean'``, ``'manhattan'`` or ``'cosine'``. If `metric` is a
        callable, it should take pairs of vectors (1D arrays) as input and, for
        each two vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    neighborhood: string or callable, optional, default: ``'nb_neighbours'``
        but can also be 'epsilon_ball'. If set to ``'nb_neighbours'``, then
        neighborhoods around points are defined by the number of points they
        include. If set to ``'epsilon_ball'``, then neighborhoods around
        points are defined by distances from the point considered, where the
        distance is computed with ``'metric'``.


    radii: tuple, optional, default: ``(10, 50)`` if ``neighborhood`` is
        ``nb_neighbours``, and ``(0.01, 0.1)`` if ``neighborhood`` is
        ``epsilon_ball``. If ``neighborhood`` is neither ``nb_neighbours``
        nor ``epsilon_ball``, then ``radii`` has to be enterred manually,
        with its first entry smaller than the second one.
        If ``neighborhood`` is ``nb_neighbours``, then radii has to
        consist of two non-negative integers, and if ``neighborhood``
        is ``epsilon_ball``, it can consist of non-negative floats.


    homology_dimensions: tuple, optional, default: ``(1, 2)``. Dimensions
        (non-negative integers) of the topological features to be detected.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.
    """

    _hyperparameters = {
        'metric': {'type': (str, FunctionType)},
        'neighborhood': {'type': str},
        'radii': {
            'type': tuple,
            'of': {'type': Real, 'in': Interval(0, np.inf, closed='left')}
            },
        'homology_dimensions': {
            'type': (tuple, list),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            }
        }

    def __init__(self, metric='euclidean', neighborhood='nb_neighbours',
                 radii=None, homology_dimensions=(1, 2), n_jobs=1):

        # metric for the point cloud
        self.metric = metric

        # type of neighborhood to consider
        self.neighborhood = neighborhood
        self.radii = radii # The default argument is None, because the user might change 'nb_neighbours'.

        # Dimensions of interest
        self.homology_dimensions = homology_dimensions

        self.n_jobs = n_jobs

        # The following object is used to compute persistence diagrams
        self.homology = VietorisRipsPersistence(
                            metric="precomputed",
                            collapse_edges=True,
                            homology_dimensions=self.homology_dimensions,
                            n_jobs=self.n_jobs)

        # The following warns if the neighborhood type is not recognized
        if self.neighborhood != 'nb_neighbours' and self.neighborhood != 'epsilon_ball':
                warnings.warn('Unrecognized neighborhood, check spelling.')

        # The following makes sure the radii parameter has been set correctly.
        if self.radii is None:
            if self.neighborhood == 'nb_neighbours':
                self.radii = (10, 50)
            elif self.neighborhood == 'epsilon_ball':
                self.radii = (0.01, 0.1)
                warnings.warn('Range parameter not entered.\
                    Empirically, radii[1] should contain at least 10 points.')
            else:
                print('neighborhood has to be either "epsilon_ball",\
                      or "nb_neighbours.')
        elif self.radii[0] > self.radii[1]:
            warnings.warn('First radius should be strictly smaller than second.\
                The values are permuted')
            self.radii = (self.radii[1], self.radii[0])
        if self.radii[1] == 0:
            warnings.warn('Second radius has to be strictly greater than 0.\
                Second radius set to 1.')
            self.radii = (self.radii[0], 1)
        elif self.radii[0] == self.radii[1]:
            warnings.warn('For meaningfull features, the first radius should\
                be strictly smaller than second.')

    def fit(self, X, y=None):
        """
        Calculates the distance matrix from the point cloud,
        unless the metric is 'precomputed' in which case it does nothing.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------

        X : ndarray of shape (nb_points, dimension)
            Input data representing  point cloud if 'metric' was not
            set to ``'precomputed'``, and a distance matrix or adjacency
            matrix of a weithed undirected graph otherwise. Can be either
            a point cloud: an array of shape ``(n_points, n_dimensions)``.
            If `metric` was set to ``'precomputed'``, then:

                - Question 2) : Should X ba allowed to be dense and sparse??

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        self._is_precomputed = self.metric == 'precomputed' # SHould we not accept precomputed anymore...
        check_array(X, accept_sparse=False)

        if self._is_precomputed:
            self.X_mat = X
        else:
            self.X = X
        self.size = len(X)
        if self.neighborhood == 'nb_neighbours' and self.size <= self.radii[0]:
            warnings.warn('First radius is too large to be relevant.\
                             Consider reducing it.')
            self.radii = (self.size-1, self.size)
        return self

    def transform(self, X):
        """
        Computes the dimension at each element of X, and returns a list of
        dimension vector for each element of X. This is done in four steps:
            - Compute the local_matrices around each point, as well as
             the indices of points in the smaller neighborhoods.
            - Construct the coned_matrix for each data point. For each
             local matrix, construct  a 'coned_matrix' which is the original
             matrix with an additional row/column with distance np.inf to the
             points in first neighborhood and distance 0 to points in second
             neighborhood.
            - Generate the persistence diagrams associated to the collection
             coned_mats, this is done using the homology object.
            - Compute the dimension vectors from the collection of diagrams
             using 'vectorizer'.

        Parameters
        ----------

        X : ndarray of shape (nb_points, dimension)
             Input data representing  point cloud if 'metric' was not set
             to ``'precomputed'``. Can be either a point
             cloud: an array of shape ``(n_points, n_dimensions)``.
             If 'self.metric' was set to ``'precomputed'``, then X is replaced
             by the data that got fitted on.


        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------

        diags : persistent diagrams # Do this better!

        """

       # check_is_fitted(self) # This causes errors...
        check_array(X, accept_sparse=False)
        loc_mats, small_neighbs = self._memberships(X) 

        self.coned_mats = Parallel(n_jobs=self.n_jobs)(
                         delayed(self._cone_off_mat)
                         (loc_mats[i], small_neighbs[i])
                         for i in range(len(X)))
        Xt = self.homology.fit_transform(self.coned_mats)
        return Xt # These are persistence diagrams

    def _memberships(self, X):
        # Computes all pairwise distances between X and training data:
        if self._is_precomputed:
        	# This might change later: Users could input columns of distances 
        	# to previous points. Could be useful if the user wants to know
        	# the local dimension around a subset of your data.
            warnings.warn('If metric is "precomputed", the input to "transform"\
                           is assumed to be the same input to "fit".')
            dist_mat = self.X_mat
        else:
            dist_mat = cdist(X, self.X, metric=self.metric)
            if dist_mat.shape[0] < dist_mat.shape[1]:
                padding = np.zeros((dist_mat.shape[1] - dist_mat.shape[0],
                                    dist_mat.shape[1]))
                dist_mat = np.vstack((dist_mat, padding))
            elif dist_mat.shape[0] > dist_mat.shape[1]:
                padding = (np.max(dist_mat) + 1) *\
                          np.ones((dist_mat.shape[0],
                                  dist_mat.shape[0] - dist_mat.shape[1]))
                dist_mat = np.hstack((dist_mat, padding))

        # Computes the membership matrix for the large neighborhood.
        # large_neighbs is a binary matrix where the ij th entry is 1
        # if the jth point is in the large neighborhood of the ith point
        # unless i==j, and is 0 otherwise.

        if self.neighborhood == 'nb_neighbours':
            large_neighbs = kneighbors_graph(
                                 dist_mat,
                                 n_neighbors=min(self.size-1, self.radii[1]),
                                 metric='precomputed', include_self=True,
                                 n_jobs=self.n_jobs).toarray()
            # Question USE CSR SCIPY?? Dense vs sparse?
        elif self.neighborhood == 'epsilon_ball':
            large_neighbs = radius_neighbors_graph(
                                 dist_mat,
                                 radius=self.radii[1],
                                 metric='precomputed',
                                 include_self=True,
                                 n_jobs=self.n_jobs).toarray()
            # Question: Is it worth it to avoice doing .toarray()?

        # Collects indices of the points in the large neighborhood.
        loc_inds = [np.nonzero(large_neighbs[i])[0] for i in range(len(X))]
        
        # Using the indices to construct the local distance matrix 
        # (with the point considered as first entry). Is this still true?
        loc_mats = [dist_mat[np.ix_(loc_inds[i], loc_inds[i])]
                             for i in range(len(X))]

        # Now looks at small neighborhood.
        # Computes the indices of small neighborhood for elements of loc_mats.
        if self.neighborhood == 'nb_neighbours':
            small_neighbs = Parallel(n_jobs=self.n_jobs)(
                             delayed(np.where)
                             (dist_mat[i][np.ix_(loc_inds[i])] <=
                              np.partition(loc_mats[i][0],
                                           self.radii[0]-1)[self.radii[0]-1],
                              1, 0)
                             for i in range(len(X)))
        elif self.neighborhood == 'epsilon_ball':
            small_neighbs = Parallel(n_jobs=self.n_jobs)(
                             delayed(np.where)
                             (dist_mat[i][np.ix_(loc_inds[i])] <=
                              self.radii[0],
                              1, 0)
                             for i in range(len(X)))
        return loc_mats, small_neighbs

    def _cone_off_mat(self, loc_mat, membership_vect):
        # Given a loc_mat (a distance matrix of size nxn), and a
        # binary membership vect of size n. Extends given matrix by one
        # to size (n+1)x(n+1) with last row having distance np.inf for
        # having a 1 in membership_vects, 0 otherwise.
        if len(loc_mat) == 1:
            warnings.warn('Second range entry chosen is too small\
                          for some points. Consider making it larger.')
        minval = 0
        maxval = np.inf

        new_row = np.where(membership_vect == 1, maxval, minval)

        new_col = np.concatenate((new_row, [0]))
        pre_augm_loc_mat = np.concatenate((loc_mat, [new_row]))
        augm_loc_mat = np.concatenate(
                                     (pre_augm_loc_mat, np.array([new_col], dtype=float).T),
                                      axis=1)
        return augm_loc_mat


    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )
