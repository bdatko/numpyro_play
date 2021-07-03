"""
Testing for pyro example importance sampling
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from numpyro_play import jax_simulate, simulate


@given(mu=arrays(float, 1, elements=floats(0,10)))
def test_simulate(mu):
    np.testing.assert_array_equal(simulate(mu[0]), jax_simulate(mu[0]))
