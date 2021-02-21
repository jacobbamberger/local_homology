import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, composite
from numpy.testing import assert_almost_equal
from scipy.sparse import coo_matrix #what is this?
from gtda.homology import VietorisRipsPersistence

from local_homology_simple import *



 
@composite
def get_dense_distance_matrices(draw):
    """Generate 2d dense square arrays of floats, with zero along the
    diagonal."""
    shapes = draw(integers(min_value=2, max_value=30))
    dm = draw(arrays(dtype=float,
                     elements=floats(allow_nan=False,
                                     allow_infinity=False,
                                     min_value=0),
                     shape=(shapes, shapes), unique=False))
    np.fill_diagonal(dm, 0)
    return dm

@composite
def radii_tuple(draw):
    nb_neighbour1 = draw(integers(min_value = 0))
    nb_neighbour2 = draw(integers(min_value = nb_neighbour1+1))
    return (nb_neighbour1, nb_neighbour2)

@composite
def test_local_homology_nb_neighbours(draw):
    """This generates a tupe (dm, nb_neighbour1, nb_neighbour2) where the first 
    entry is a distance matrix, the other two are integers and nb_neighbour1<nb_neighbour2<len(dm)+1"""
    shapes = draw(integers(min_value=2, max_value=30))
    dm = draw(arrays(dtype=float,
                     elements=floats(allow_nan=False,
                                     allow_infinity=False,
                                     min_value=0),
                     shape=(shapes, shapes), unique=False))
    np.fill_diagonal(dm, 0)

    nb_neighbour1 = draw(integers(min_value = 0, max_value=len(dm)))
    nb_neighbour2 = draw(integers(min_value = nb_neighbour1+1, max_value=len(dm)+1))
    radii=(nb_neighbour1, nb_neighbour2)
    return (dm, radii)



#@pytest.mark.parametrize('neighborhood', ["nb_neighbours", "epsilon_ball"])
#@pytest.mark.parametrize('radii' = st.tuples(st.integers(), st.integers()))
#@settings(deadline=500)
#nb_neighbour1 = draw(integers(min_value=0, max_value=len(dm)-1)) 
@settings(max_examples=500)
@given(dm_radii=test_local_homology_nb_neighbours())#, nb_neighbour1 = integers(min_value=0, max_value=len(dm)), nb_neighbour2 = integers(min_value=nb_neighbour1))
def test_local_homology_nb_neighbours(dm_radii):#, nb_neighbour1, nb_neighbour2):
    print('testing')
    (dm, radii)=dm_radii
    #nb_neighbour1 = draw(integers(min_value = 0, max_value=len(dm)))
    #nb_neighbour2 = draw(integers(min_value = nb_neighbour1+1))
    #assert nb_neighbour1 <= len(dm)
    #assert nb_neighbour1 < nb_neighbour2 
    lh = local_homology(metric="precomputed", radii=radii)
    lh.fit(dm)
    lh.transform(dm)
    print('testing local nb_neighbours')

