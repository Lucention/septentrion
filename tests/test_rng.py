import numpy as np

from septentrion.rng import set_seed


def test_set_seed_returns_generator():
    rng = set_seed(123)
    assert isinstance(rng, np.random.Generator)


def test_set_seed_is_reproducible():
    a = set_seed(123).random(5)
    b = set_seed(123).random(5)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = set_seed(1).random(5)
    b = set_seed(2).random(5)
    assert not np.array_equal(a, b)
