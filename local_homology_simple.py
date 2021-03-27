from numbers import Real
from types import FunctionType

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist

from gtda.homology import VietorisRipsPersistence
# Plotting functions
from gtda.plotting import plot_diagram
#nearest neighbours:
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
# Vectorize functions:
from gtda.diagrams import features

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin, OutlierMixin, BaseEstimator, \
                         TransformerMixin
from sklearn.metrics import pairwise_distances

import umap

from joblib import Parallel, delayed

import pandas as pd
import plotly.express as px

import warnings

from gtda.diagrams.features import PersistenceEntropy
from gtda.diagrams.preprocessing import Filtering

from gtda.utils.intervals import Interval 
from gtda.utils.validation import validate_params, check_point_clouds





class local_homology(BaseEstimator, TransformerMixin): #plotterMixin?
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

    neighborhood: string or callable, optional, default: ``'nb_neighbours'`` but
    	can also be 'epsilon'.
    	If set to ``'nb_neighbours'``, then neighborhoods around points are defined 
    	by the number of points they include.
    	If set to ``'epsilon_ball'``, then neighborhoods around points are defined by distances
    	from the point considered.


    neighborhood_params: Dictionary, optional, default: {nb_neighbours1 : 10, nb_neighbours2: 50}.
       	If neighborhood above is set to  is ``'nb_neighbours'``, then neighborhood_params needs two keys 
       	``'nb_neighbours1'`` and ``'nb_neighbours2'`` defining the smaller neighborhood and the 
       	larger neighborhood to be considered around each point.
    	If neighborhood above is set to ``'epsilon'``,  then neighborhood_params needs two keys, 
       	``'epsilon1'`` and ``'epsilon2'`` defining the smaller neighborhood and the 
       	larger neighborhood to be considered around each point.

    homology: persistence object, optional, default: VietorisRipsPersistence.
    	Any instance of an object inputing a list of point clouds and outputting 
   		persistence diagrams when transformed. (Any object from simplicial.py)

    homology_parameters: Dictionary of hyperparameters to initialize 'homology'.

    vectorizer: transformer, optional, default: modified_Persistence_Entropy.
    	Any transformer that inputs a list of persistence diagrams and outputs a list of vector
    	of the same size as the list of diagrams.

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
		#'homology': {'type': } how do I check for transformers???


	def __init__(self, metric = 'euclidean', neighborhood = 'nb_neighbours', radii=None, homology_dimensions=None,
		homology=None, vectorizer=None, n_jobs=None):

		#metric of general point cloud
		self.metric = metric

		#type of neighborhood to consider
		self.neighborhood = neighborhood
		self.radii = radii

		self.homology_dimensions = homology_dimensions

		#persistence homology tool
		self.homology = homology 


		#vectorizing persistence diagrams
		self.vectorizer = vectorizer
		self.modified_persistence = False

		
		self.check_is_fitted = False
		self.check_is_transformed = False
		self.dgms = None

		self.n_jobs = n_jobs

		if self.homology_dimensions is None:
			self.homology_dimensions = (0, 1, 2,)

		if self.homology is None:
			self.homology = VietorisRipsPersistence(metric="precomputed", collapse_edges=True, homology_dimensions=self.homology_dimensions, n_jobs=self.n_jobs)
		else:
			self.homology_dimensions = self.homology.homology_dimensions 

		if self.radii is None:
			if self.neighborhood == 'nb_neighbours':
				self.radii= (10, 50)
			elif self.neighborhood == 'epsilon_ball':
				self.radii = (0.01, 0.1)
				warnings.warn('Range parameter not entered. Empirically, range[1] should encircle at least 10 points.')
			#else:
			#	print('User defined neighborhood needs user defined hyperparameters.')
		elif self.radii[0]>self.radii[1]:
			warnings.warn('First radius should be strictly smaller than second. The values are permuted')
			self.radii = (self.radii[1], self.radii[0])
		if self.radii[1]==0:
			warnings.warn('Second radius has to be strictly greater than 0. Second radius set to 1.')
			self.radii=(self.radii[0], 1)
		#elif self.radii[0] == self.radii[1]:
		#	warnings.warn('First radius should be strictly smaller than second.') #what to do in this case?




		if self.vectorizer is None:
			self.vectorizer = PersistenceEntropy()
			self.modified_persistence = True




	def fit(self, X, y=None):
		"""
		Calculates the distance matrix from the point cloud, unless the metric is 'precomputed'
		in which case it does nothing.

		This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------

        X : ndarray of shape (nb_points, dimension)
        	Input data representing  point cloud if 'metric' was not set to ``'precomputed'``,
        	and a distance matrix or adjacency matrix of a weithed undirected graph otherwise.
        	Can be either a point cloud: an array of shape ``(n_points, n_dimensions)``. If 
        	`metric` was set to ``'precomputed'``, then:

        		- ?? Something with X dense/sparse/filtration??

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

		"""
		#validate_params(	
		#	self.get_params(), self._hyperparameters, exclude=['homology', 'homology_params', 'vectorizer', 'n_jobs'])

		validate_params(
			self.get_params(), self._hyperparameters, exclude=['homology_dimensions', 
			'homology', 'vectorizer', 'homology__coeff', 'collapse_edges', 'homology__collapse_edges',
			'homology__homology_dimensions', 'homology__infinity_values', 'homology__max_edge_length',
			'homology__metric', 'homology__n_jobs', 'homology__reduced_homology',
			'vectorizer__n_jobs', 'vectorizer__nan_fill_value', 'vectorizer__normalize',
			'n_jobs'])
		self._is_precomputed = self.metric == 'precomputed'
		check_point_clouds(np.array([X]), accept_sparse=False, #is there a check_point_cloud?
			distance_matrices=self._is_precomputed)



		if self.metric == 'precomputed':
			self.X_mat = np.array(X)
		else:
			self.X = np.array(X)
			self.X_mat = squareform(pdist(X, metric=self.metric))
			#pairwise_distances(X, metric = self.metric) #CHANGE THIS
		self.check_is_fitted = True
		self.size = len(X)
		if self.neighborhood == 'nb_neighbours' and self.size <= self.radii[0]:
			warnings.warn('First radius is too large to be relevant. Consider reducing it.')
			self.radii=(self.size-1, self.size)
		if self.neighborhood =='epsilon_ball' and np.max(self.X_mat)<=self.radii[0]:
			warnings.warn('First radius is too large to be relevant. Consider reducing it.')
		return self

	def transform(self, X):
		"""
		For each point x in the distance matrix, compute the dimension vector  at that point.
		This is done in four steps:
			- Compute the local_matrices around each point, as well as the indices of points
			 in the smaller neighborhoods.
			- Construct the coned_matrix for each data point. For each local matrix, construct 
			 a 'coned_matrix' which is the original matrix with an additional row/column with 
			 distance np.inf to the points in first neighborhood and distance 0 to points in 
			 second neighborhood.
			- Generate the persistence diagrams associated to the collection coned_mats, this 
			 is done using the homology object.
			- Compute the dimension vectors from the collection of diagrams using the vectorizer
			 object.
	
		Parameters
        ----------

        X : ndarray of shape (nb_points, dimension)
        	Input data representing  point cloud if 'metric' was not set to ``'precomputed'``,
        	and a distance matrix or adjacency matrix of a weithed undirected graph otherwise.
        	Can be either a point cloud: an array of shape ``(n_points, n_dimensions)``. If 
        	`metric` was set to ``'precomputed'``, then:

        		- ?? Something with X dense/sparse/filtration??


        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        dim_vects: ndarray of shape (nb_points, local_dimensions)
        			An element is a vector where the ith entry is the value corresponding to
        			the ith dimension's considered.

		"""

		if self.check_is_fitted: #check that transform on same X as before? 
			self.check_is_transformed = True
			loc_mats, small_neighbs = self._memberships(X)

			self.coned_mats = Parallel(n_jobs=self.n_jobs)(delayed(self._cone_off_mat)
				(loc_mats[i], small_neighbs[i]) for i in range(len(X)) ) 
			self.diags = self.homology.fit_transform(self.coned_mats) 

			if self.modified_persistence:
				self.dim_vects = 2**self.vectorizer.fit_transform(self.diags)
			else:
				self.dim_vects = self.vectorizer.fit_transform(self.diags)
		else:
			print('Tried to transform before fitting.')

		return np.hstack((X, self.dim_vects))


    
	def _memberships(self, X):
		#First compute the membership matrix for the large neighborhood. large_neighbs is a binary matrix
		#where the ij th entry is 1 iff the jth point is in the large neighborhood of the ith point unless i=j.
		if self._is_precomputed:
			dist_mat = self.X_mat
		else:
			dist_mat = cdist(X, self.X, metric=self.metric)
			if dist_mat.shape[0] < dist_mat.shape[1]:
				padding = np.zeros((dist_mat.shape[1]-dist_mat.shape[0], dist_mat.shape[1]))
				dist_mat = np.vstack((dist_mat, padding))
			elif dist_mat.shape[0] > dist_mat.shape[1]:
				padding = (np.max(dist_mat)+1)*np.ones((dist_mat.shape[0], dist_mat.shape[0]-dist_mat.shape[1]))
				dist_mat = np.hstack((dist_mat, padding))


		if self.neighborhood == 'nb_neighbours':
			large_neighbs = kneighbors_graph(dist_mat, n_neighbors=min(self.size-1, self.radii[1]), metric='precomputed', include_self=True, n_jobs=self.n_jobs).toarray() #USE CSR SCIPY
		elif self.neighborhood == 'epsilon_ball':
			large_neighbs = radius_neighbors_graph(dist_mat, radius=self.radii[1], metric='precomputed', include_self=True, n_jobs=self.n_jobs).toarray()
		else:
			warnings.warn('Unrecognized neighbourhood method, using default.')
			large_neighbs = kneighbors_graph(dist_mat, n_neighbors=min(self.size-1, 50) , metric='precomputed', include_self=True, n_jobs=self.n_jobs).toarray() #is this line already covered?

		#Collect indices of the points in the large neighborhood. Construct the local distance matrix (with the point considered as first entry).
		#loc_inds = [np.concatenate(([i], np.nonzero(large_neighbs[i])[0]), axis=0) for i in range(len(X))] #PARALLELIZE, includeself??!!!
		loc_inds = [np.nonzero(large_neighbs[i])[0] for i in range(len(X))] #calling np.nonzero gives a mesh already....
		loc_mats = np.array([self.X_mat[np.ix_(loc_inds[i], loc_inds[i])] for i in range(len(X))]) #PARALLELIZE



		#Now looks at small neighborhood. Computes the indices of small neighborhood for each element of loc_mats.
		"""if self.neighborhood == 'nb_neighbours':
			small_neighbs = Parallel(n_jobs=self.n_jobs)(delayed(np.where)
				(loc_mat[0] <= np.partition(loc_mat[0], self.radii[0]-1)[self.radii[0]-1], 1, 0) for loc_mat in loc_mats ) #[np.where(loc_mat[0]<np.partition(loc_mat[0], self.neighborhood_params['nb_neighbours'])[self.neighborhood_params['nb_neighbours']], 1, 0) for loc_mat in loc_mats] #PARALLELIZE THIS
		elif self.neighborhood == 'epsilon_ball':
			small_neighbs = Parallel(n_jobs=self.n_jobs)(delayed(np.where)
				(loc_mat[0] <= self.radii[0], 1, 0) for loc_mat in loc_mats ) #[np.where(loc_mat[0] < self.neighborhood_params['epsilon'], 1, 0) for loc_mat in loc_mats] #1 is inside small neighborhood, 0 if not. What about point itself? seems like it is
		else:
			warnings.warn('Unrecognized neighbourhood method, using default.')
			large_neighbs = kneighbors_graph(self.X_mat, n_neighbors=10, metric='precomputed', include_self=False, n_jobs=self.n_jobs).toarray()"""

		if self.neighborhood == 'nb_neighbours':
			small_neighbs = Parallel(n_jobs=self.n_jobs)(delayed(np.where)
				(dist_mat[i][np.ix_[loc_inds[i]]] <= np.partition(loc_mat[0], self.radii[0]-1)[self.radii[0]-1], 1, 0) for i in range(len(X)) )
		elif self.neighborhood == 'epsilon_ball':
			small_neighbs = Parallel(n_jobs=self.n_jobs)(delayed(np.where)
				(dist_mat[i][np.ix_(loc_inds[i])] <= self.radii[0], 1, 0) for i in range(len(X)))

		return loc_mats, small_neighbs



	def _cone_off_mat(self, loc_mat, membership_vect, n=2):
		#Given loc_mat, a distance matrix of size nxn, and binary membership vect of size n. Extends given matrix by one
		# to size (n+1)x(n+1) with last row having distance np.inf for having a 1 in membership_vects, 0 otherwise.
		if len(loc_mat) == 1:
			warnings.warn('Second range entry chosen is too small for some points. Consider making it larger.')
		minval = 0 
		maxval = np.inf 

		new_row = np.where(membership_vect == 1, maxval, minval)

		new_col = np.concatenate((new_row, [0]))
		pre_augm_loc_mat = np.concatenate((loc_mat, [new_row]))
		augm_loc_mat = np.concatenate((pre_augm_loc_mat, np.array([new_col]).T), axis=1)
		return augm_loc_mat


	def plot(self, X, dimension = None):
		if not self.check_is_transformed:
			self.fit_transform(X)
		if not(dimension is None) and dimension in self.homology.homology_dimensions:
			color = self.dim_vects[:, np.argwhere(np.array(self.homology.homology_dimensions)==dimension)[0][0]]
		else: 
			print('invalid dimension, colouring with dimension '+ str(self.homology.homology_dimensions[0]))
			color = self.dim_vects[:, 0]
		return self.plot_point_cloud(np.array(X), color)

	def plot_point_cloud(self, ndarr, color = None):
		if ndarr.shape[1]>=3:
			df_plot = pd.DataFrame(np.array(ndarr), columns = ["x","y","z"])
			return px.scatter_3d(df_plot, x="x", y="y", z="z", color = color)
		elif ndarr.shape[1]==2:
			df_plot = pd.DataFrame(np.array(ndarr), columns = ["x","y"])
			return px.scatter(df_plot, x="x", y="y", color = color)
		else:
			print("Cannot plot...")


"""	def predict(self, X, use_test=False):
		# check use_test. If false then prepare the large distance matrix setting inf to the test x test distances
		if self.precomputed:
			warning.warn('Cannot predict with precomputed data.')
		if not use_test:
			cdist(X, self.X, self.metric)"""

