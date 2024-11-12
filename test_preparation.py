""" This module runs the unit test for shift FFT data """
import unittest
import numpy as np
import pandas as pd
import preparation as prep  # Assuming you have a module called 'preparation'

# Dummy test function (assuming you have a function 'dummy' in 'preparation' module)
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
        self.assertNotEqual(shifted_fft[center_index, center_index], 0, "Center frequency should not be zero.")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
