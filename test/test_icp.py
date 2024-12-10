import sys
import os

import numpy as np

# Add the parent directory to the Python path to import icp.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from icp import icp

def test_icp_identity():
    source = np.array([[0, 0], [1, 0], [0, 1]])
    target = source.copy()

    transformed, transformation = icp(source, target)
    assert np.allclose(transformed, target, atol=1e-6)
    assert np.allclose(transformation["rotation"], np.eye(2), atol=1e-6)
    assert np.allclose(transformation["translation"], np.zeros(2), atol=1e-6)

def test_icp_translation():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    target = source + np.array([1, 2, 0])

    transformed, transformation = icp(source, target)
    assert np.allclose(transformed, target, atol=1e-6)
    assert np.allclose(transformation["translation"], [1, 2, 0], atol=1e-1)

if __name__ == "__main__":
    test_icp_identity()
    test_icp_translation()
    test_icp_rotation()
    print("All tests passed!")
