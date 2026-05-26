from septentrion.geo.distance import haversine_m


def test_zero_distance():
    assert haversine_m(0.0, 0.0, 0.0, 0.0) == 0.0


def test_one_degree_latitude_is_about_111km():
    d = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert 110_000 < d < 112_000
