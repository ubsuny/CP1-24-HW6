"""
This module contains unit tests and fixtures for testing the preparation module.
It includes functions for testing FFT transformations, windowing, and loading data.
"""

import os
import pickle
import unittest

import pytest
import numpy as np
import pandas as pd

import preparation as prep
from preparation import apply_2d_windowing, remove_2d_windowing


def test_dummy():
    """Unit test for dummy function"""
    assert prep.dummy() == 0


def shift_fft(image_df):
    """
    Takes the 2D-FFT of an image represented by a pandas DataFrame and shifts the result
    so that the zero frequency component is in the center of the image.
    """
    image_array = image_df.to_numpy()
    fft_result = np.fft.fft2(image_array)
    shifted_fft = np.fft.fftshift(fft_result)
    shifted_fft_df = pd.DataFrame(shifted_fft, index=image_df.index, columns=image_df.columns)
    return shifted_fft_df


class TestFFTShift(unittest.TestCase):
    """
    Unit test case for the shift_fft function, which applies a 2D FFT to an image
    and shifts the zero frequency component to the center.
    """

    def setUp(self):
        """
        Set up a simple 2D image to test the FFT shift function.
        """
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
        """
        shifted_fft_df = shift_fft(self.image_df)
        shifted_fft = shifted_fft_df.to_numpy()
        center_index = len(self.image_df) // 2
        self.assertNotEqual(shifted_fft[center_index, center_index], 0)


@pytest.fixture(name="square_image")
def setup_square_image():
    """Fixture that returns a 2D square-shaped structure of data."""
    pix = 256
    data = np.random.randint(256, size=(pix, pix), dtype=np.uint8)
    return pd.DataFrame(data)


@pytest.fixture(name="rectangular_image")
def setup_rectangular_image():
    """Fixture that returns a 2D rectangular-shaped structure of data."""
    pix_x, pix_y = 512, 256
    data = np.random.randint(256, size=(pix_x, pix_y), dtype=np.uint8)
    return pd.DataFrame(data)


@pytest.fixture(name="image_data_fixture")
def image_data_fixture():
    """Fixture providing a sample 2D numpy array."""
    return np.random.rand(100, 100)


@pytest.mark.parametrize("window_type", ["hann", "hamming", "gaussian"])
def test_apply_2d_windowing_multiple_windows(image_data_fixture, window_type):
    """
    Test the apply_2d_windowing function with multiple window types.
    """
    windowed_data, window = apply_2d_windowing(image_data_fixture, window_type)
    assert windowed_data.shape == image_data_fixture.shape
    assert window.shape == image_data_fixture.shape


def test_remove_2d_windowing(image_data_fixture):
    """
    Test the remove_2d_windowing function to ensure it correctly reverts windowed data.
    """
    windowed_data, window = apply_2d_windowing(image_data_fixture)
    unwindowed_data = remove_2d_windowing(windowed_data, window)
    center = slice(image_data_fixture.shape[0] // 4, 3 * image_data_fixture.shape[0] // 4)
    central_original = image_data_fixture[center, center]
    central_unwindowed = unwindowed_data[center, center]
    assert np.allclose(central_original, central_unwindowed, atol=1e-2)


@pytest.fixture(name="sample_pickle_fixture")
def setup_sample_pickle():
    """
    Fixture to set up and tear down a sample pickle file for testing.
    """
    pickle_path = "images/test_img_data.pkl"
    os.makedirs("images", exist_ok=True)
    sample_data = pd.DataFrame([[1, 2], [3, 4]])
    with open(pickle_path, 'wb') as file:
        pickle.dump(sample_data, file)
    yield sample_data
    os.remove(pickle_path)


@pytest.fixture(name="empty_pickle_fixture")
def setup_empty_pickle():
    """
    Fixture to set up an empty pickle file for testing.
    """
    pickle_path = "images/test_img_data.pkl"
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    empty_data = pd.DataFrame()
    with open(pickle_path, 'wb') as file:
        pickle.dump(empty_data, file)
    yield pickle_path
    if os.path.isfile(pickle_path):
        os.remove(pickle_path)


def test_load_pickle_data_with_sample_data(sample_pickle_fixture):
    """
    Tests load_pickle_data with the sample data to ensure it matches the expected structure.
    """
    pickle_path = "images/test_img_data.pkl"
    df_default = prep.load_pickle_data(pickle_path)
    pd.testing.assert_frame_equal(df_default, sample_pickle_fixture)


def test_load_pickle_data_with_empty_data(empty_pickle_fixture):
    """
    Test loading an empty pickle file to ensure it returns an empty DataFrame.
    """
    df_empty = prep.load_pickle_data(empty_pickle_fixture)
    assert df_empty.empty


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
