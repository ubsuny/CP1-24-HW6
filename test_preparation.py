""" This is to test the functions created in the preparation module """

import numpy as np
import pandas as pd
import pytest
import preparation as prep
# pylint: disable=W0621

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

def test_twod_fft_mag(sq_img):
    """this test makes sure that the twod_fft_mag functions properly, by testing this also makes
    sure that the length and widths of these two data sets are the same with the same assert"""
    img = sq_img
    compare = np.isclose(prep.twod_fft_mag(img),np.fft.fft2(np.array(img)), atol=1e-8)
    assert np.all(compare)

def test_twod_inv_fft(sq_img):
    """this compares the inves of the fft function agenced the original unlaltered
    2d matrix this also makes sure that the length and widths of these two data
    sets are the same with the same assert"""
    img = sq_img
    compare1 = np.isclose(np.real( prep.twod_inv_fft( prep.twod_fft_mag( img ))),
                          np.array(img), atol=1e-7)
    assert np.all(compare1)

def test_twod_calc_freq(sq_img):
    """This test just makes sure that what the function outputs is indeed the correct size"""
    img = sq_img
    array1, array2 = prep.twod_calc_freq(img,1,1)
    image = np.array(img)
    width, height = image.shape
    assert len(array1) == width
    assert len(array2) == height
