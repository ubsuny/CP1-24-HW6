""" This is to test the functions created in the preparation module """

import numpy as np
import pandas as pd
import pytest
import preparation as prep

def test_dummy():
    """ unit test for dummy function """
    assert prep.dummy() == 0

@pytest.fixture
def sq_img():
    """ Fixture that returns a 2D square-shaped structure of data
        in pandas DataFrame format (e.g., 256x256) with default indexing. """
    # Define size
    pix = 256
    # Generate random pixel intensities for the simulated image (grayscale representation)
    data = np.random.randint(256, size = (pix, pix), dtype = np.uint8)
    # Convert to DataFrame
    return pd.DataFrame(data)

@pytest.fixture
def rect_img():
    """ Fixture that returns a 2D rectangular-shaped structure of data
        in pandas DataFrame format (e.g., 512x256) with default indexing. """
    # Define size
    pix_x = 512
    pix_y = 256
    # Generate random pixel intensities for the simulated image (grayscale representation)
    data = np.random.randint(256, size = (pix_x, pix_y), dtype = np.uint8)
    # Convert to DataFrame
    return pd.DataFrame(data)
