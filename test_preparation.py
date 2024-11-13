
""" This module runs the unit test for shift FFT data """
import unittest
import pytest
import numpy as np
import pandas as pd
import preparation as prep
# pylint: disable=W0621

def test_dummy():
    """ Unit test for dummy function """
    assert prep.dummy() == 0

# Assuming 'shift_fft' is your function that does the FFT shift
def shift_fft(image_df):
    """
    Takes the 2D-FFT of an image represented by a pandas DataFrame and shifts the result
    so that the zero frequency component is in the center of the image.
    """
    image_array = image_df.to_numpy()

    # Perform the 2D FFT
    fft_result = np.fft.fft2(image_array)

    # Shift the zero frequency component to the center
    shifted_fft = np.fft.fftshift(fft_result)

    # Convert to pandas DataFrame for consistency with input
    shifted_fft_df = pd.DataFrame(shifted_fft, index=image_df.index, columns=image_df.columns)

    return shifted_fft_df


class TestFFTShift(unittest.TestCase):
    """
    Unit test case for the shift_fft function, which applies a 2D FFT to an image
    and shifts the zero frequency component to the center.
    
    This class tests:
    - The shape of the output is the same as the input.
    - The output is a pandas DataFrame.
    - The center of the FFT result contains a non-zero value after the shift.
    """
    def setUp(self):
        """
        Set up a simple 2D image to test the FFT shift function.
        """
        # Create a small 2D image with a known simple pattern (e.g., a 4x4 grid)
        self.image_df = pd.DataFrame([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])

    def test_fft_shift_shape(self):
        """
        Test if the FFT shift function returns a DataFrame of the same shape.
        """
        shifted_fft_df = shift_fft(self.image_df)
        self.assertEqual(shifted_fft_df.shape, self.image_df.shape)

    def test_fft_shift_type(self):
        """
        Test if the FFT shift function returns a pandas DataFrame.
        """
        shifted_fft_df = shift_fft(self.image_df)
        self.assertIsInstance(shifted_fft_df, pd.DataFrame)

    def test_fft_shift_center(self):
        """
        Test that the zero frequency component is shifted to the center.
        For a very simple pattern, we check that the center is non-zero.
        """
        shifted_fft_df = shift_fft(self.image_df)
        shifted_fft = shifted_fft_df.to_numpy()

        # Check the center element of the shifted FFT (should be non-zero)
        center_index = len(self.image_df) // 2
        self.assertNotEqual(shifted_fft[center_index, center_index], 0,
                            "Center frequency should not be zero.")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

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
