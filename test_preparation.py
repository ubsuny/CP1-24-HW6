""" This is to test the functions created in the preparation module """

import os
import pickle
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
