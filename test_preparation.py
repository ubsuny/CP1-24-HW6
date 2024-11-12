""" unit test template """

import preparation as prep
import numpy as np
import pandas as pd
from PIL import Image
import unittest

def test_dummy():
    """ unit test for dummy function """
    assert prep.dummy() == 0
    
# Get current image dimensions
    rows, cols = image_data.shape
    print(f"Image dimensions: {rows} rows, {cols} columns")

    # Convert the image to a NumPy array
    img_data_array = np.array(img)

    # Check the shape of the array (height, width, channels)
    print(img_data_array.shape)

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

    # Run the tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
