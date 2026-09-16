"""Unit tests for perspective dewarping module."""

import numpy as np
from src.preprocessing.dewarp import order_points, dewarp_card


def test_order_points():
    pts = np.array([[100, 200], [10, 20], [10, 200], [100, 20]], dtype="float32")
    ordered = order_points(pts)
    assert np.array_equal(ordered[0], [10, 20])   # top-left
    assert np.array_equal(ordered[1], [100, 20])  # top-right
    assert np.array_equal(ordered[2], [100, 200]) # bottom-right
    assert np.array_equal(ordered[3], [10, 200])  # bottom-left


def test_dewarp_card_fallback():
    # Empty or uniform image should safely return unchanged
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = dewarp_card(img)
    assert result.shape == img.shape

    # None should return None
    assert dewarp_card(None) is None
