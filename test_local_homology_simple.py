import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, composite, lists
from numpy.testing import assert_almost_equal
from scipy.sparse import coo_matrix #what is this?
from gtda.homology import VietorisRipsPersistence
from scipy.spatial.distance import squareform

from local_homology_simple import *







"""@given(dims=gen_dimensions(), dm_radii=gen_local_homology_epsilon_ball())
def test_dimensions(dims, dm_radii):
    (dm, radii)=dm_radii
    lh = local_homology(metric="precomputed", neighborhood='epsilon_ball', radii=radii, homology_dimensions=dims)
    lh.fit(dm)
    lh.transform(dm)"""





#______________________________________

@composite
def gen_3d_point_cloud(draw):
    three_floats = lists(floats(min_value=0, max_value=1), min_size=3, max_size=3, unique=True)
    return np.array(draw(lists(three_floats, min_size=2, max_size=20))) #No Size one cloud!!!

@composite
def gen_dimensions(draw):
    return tuple(draw(lists(integers(min_value=0, max_value=10), min_size=1, max_size=5, unique=True)))

@composite
def gen_epsilon_radii(draw):
    epsilon1 = draw(floats(min_value = 0, max_value=1))
    epsilon2 = draw(floats(min_value = 0, max_value=1))
    return (epsilon1, epsilon2)

@composite 
def gen_nb_neighbours_radii(draw):
    nb_neighbour1 = draw(integers(min_value = 1, max_value=20))
    nb_neighbour2 = draw(integers(min_value = nb_neighbour1+1, max_value=21))
    return (nb_neighbour1, nb_neighbour2)



@given(point_cloud=gen_3d_point_cloud(), dims=gen_dimensions(), epsilon_radii=gen_epsilon_radii(), nb_neighbours_radii=gen_nb_neighbours_radii())
@pytest.mark.parametrize("neighborhood", ["epsilon_ball", "nb_neighbours"])#, "giberish"])
@pytest.mark.parametrize("metric", ["precomputed", "euclidean"])
def test_lh_transform(metric, neighborhood, point_cloud, dims, epsilon_radii, nb_neighbours_radii):
    if neighborhood == "epsilon_ball":
        radii = epsilon_radii
    elif neighborhood == "nb_neighbours":
        radii = nb_neighbours_radii
    if metric == 'precomputed':
        X = squareform(pdist(point_cloud))
    else:
        X = point_cloud

    lh = local_homology(metric=metric, neighborhood=neighborhood, radii=radii, homology_dimensions=dims)
    lh.fit(X)
    lh.transform(X)
    #lh.plot(X)


    lh2 = local_homology(metric=metric, neighborhood=neighborhood, radii=radii, homology_dimensions=dims)
    lh2.fit_transform(X)
    #lh2.plot(X, dimension=0)

    #lh3 = local_homology(metric=metric, neighborhood=neighborhood, radii=radii, homology_dimensions=dims)
    #lh3.plot(X, dimension = 1)


