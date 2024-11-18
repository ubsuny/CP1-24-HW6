""" This is to test the functions created in the preparation module """

import os
import pickle
import unittest
import pytest
import numpy as np
import pandas as pd
import preparation as prep


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

def test_averager():
    """
    test_averager verifies that the averager
    function can correctly calculate the 
    average value within a 2D list
    """
    x=[1,2,3,4,-1,-2,-3,-4]
    y=[]
    i=0
    while i<5:
        y.append(x)
        i+=1
    assert prep.averager(y)==2.5

def test_erase():
    """
    test_erase verifies that the values above the 
    input value are removed from a 2D list. 
    """
    x=[67.999,2,100,9,-2-77j,-50,-30,-4, 2+4j,9+3j]
    y=[]
    for i in range(5):
        y.append(x)
    ave=prep.averager(y)
    z=prep.erase(2.5,y)
    for i in z:
        for j in i:
            assert j<=ave

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

@pytest.fixture(name='sq_img')
def setup_sq_img():
    """ Fixture that returns a 2D square-shaped structure of data
        in pandas DataFrame format (e.g., 256x256) with default indexing. """
    # Define size
    pix = 256
    # Generate random pixel intensities for the simulated image (grayscale representation)
    data = np.random.randint(256, size = (pix, pix), dtype = np.uint8)
    # Convert to DataFrame
    return pd.DataFrame(data)

@pytest.fixture(name='rect_img')
def setup_rect_img():
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


# Path to the test pickle file
PICKLE_PATH = "images/test_img_data.pkl"

@pytest.fixture(name="sample_pickle_fixture")
def setup_sample_pickle():
    """
    Fixture to set up and tear down a sample pickle file for testing.
    Creates a 2x2 DataFrame, saves it to the specified file path, and removes it after the test.
    """
    os.makedirs("images", exist_ok=True)  # Ensure the directory exists
    sample_data = pd.DataFrame([[1, 2], [3, 4]])
    with open(PICKLE_PATH, 'wb') as file:
        pickle.dump(sample_data, file)
    yield sample_data
    os.remove(PICKLE_PATH)

@pytest.fixture(name="custom_structure_fixture")
def setup_custom_structure_fixture():
    """ Fixture for custom data with a specific structure size """
    data = {'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}  # 2 rows and 3 columns
    df = pd.DataFrame(data)
    # Adjust index and columns based on the structure size (2x3)
    df.index = [0, 1]  # Integer index for 2 rows
    df.columns = [0, 1, 2]  # Integer columns for 3 columns
    with open(PICKLE_PATH, 'wb') as file:
        pickle.dump(df, file)
    yield df
    os.remove(PICKLE_PATH)

@pytest.fixture(name="empty_pickle_fixture")
def setup_empty_pickle():
    """
    Fixture to set up and tear down an empty DataFrame pickle file for testing.
    Creates an empty DataFrame, saves it to the specified file path, and removes it after the test.
    """
    empty_data = pd.DataFrame()
    # Ensure the directory exists
    os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)

    # Save the empty DataFrame to a pickle file
    with open(PICKLE_PATH, 'wb') as file:
        pickle.dump(empty_data, file)

    yield PICKLE_PATH  # Yield the file path for loading in tests

    # After the test, remove the pickle file
    if os.path.isfile(PICKLE_PATH):
        os.remove(PICKLE_PATH)

def test_load_pickle_data_with_sample_data(sample_pickle_fixture, capsys):
    """
    Tests load_pickle_data with the sample data to ensure it matches the expected 2x2 structure.
    """
    df_default = prep.load_pickle_data(PICKLE_PATH)

    # Capture printed output
    captured = capsys.readouterr()

    # Ensure that the DataFrame matches the sample data
    pd.testing.assert_frame_equal(df_default, sample_pickle_fixture)

    # Ensure the index and columns match the original size and format
    assert df_default.index.tolist() == sample_pickle_fixture.index.tolist()
    assert df_default.columns.tolist() == sample_pickle_fixture.columns.tolist()

    # Check printed output for errors or warnings
    assert "Warning: The loaded DataFrame is empty." not in captured.out
    assert "Error: The file 'images/non_existent_file.pkl' was not found." not in captured.out

def test_load_pickle_data_with_custom_structure(custom_structure_fixture, capsys):
    """
    Test loading a pickle file with a custom structure size (2x3).
    """
    # Load data using the function with a custom structure (2x3)
    df_default = prep.load_pickle_data(PICKLE_PATH, structure_size=(2, 3))

    # Capture printed output
    captured = capsys.readouterr()

    # Ensure the loaded DataFrame matches the custom structure fixture (2x3)
    pd.testing.assert_frame_equal(df_default, custom_structure_fixture)

    # Ensure the index and columns match the expected format
    assert df_default.index.tolist() == custom_structure_fixture.index.tolist()
    assert df_default.columns.tolist() == custom_structure_fixture.columns.tolist()

    # Check for warnings or errors in captured output
    assert "Warning: The loaded DataFrame is empty." not in captured.out
    assert "Error: The file 'images/non_existent_file.pkl' was not found." not in captured.out

def test_load_pickle_data_with_empty_data(empty_pickle_fixture, capsys):
    """
    Test loading an empty pickle file to ensure it returns an empty DataFrame
    and prints the appropriate warning.
    """
    # Load the empty data pickle file
    df_empty = prep.load_pickle_data(empty_pickle_fixture)

    # Capture printed output
    captured = capsys.readouterr()

    # Check if the returned DataFrame is empty
    assert df_empty.empty

    # Ensure that the warning message is printed
    assert "Error loading data: The loaded DataFrame is empty." in captured.out

def test_load_pickle_data_with_nonexistent_file(capsys):
    """
    Test loading a non-existent pickle file to ensure it prints the correct error message.
    """
    pickle_file_path = 'nonexsiting/file/path/invalid_file.pkl'
    # Try to load the non-existent file
    df_non_existent = prep.load_pickle_data(pickle_file_path)

    # Capture printed output
    captured = capsys.readouterr()

    # Check if the returned DataFrame is empty
    assert df_non_existent.empty

    # Ensure that the error message is printed
    assert "Error loading data: The file '{pickle_file_path}' was not found." in captured.out
