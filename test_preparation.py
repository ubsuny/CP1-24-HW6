"""
Unit Test Module for preparation.py
"""

import os
import pickle
import pytest
import pandas as pd
import preparation as prep

# Path to the test pickle file
PICKLE_PATH = "test_periodic_structure.pkl"

@pytest.fixture(name='pickle_file')
def setup_pickle_file():
    """
    Fixture to set up and tear down a sample pickle file for testing.
    This creates a 2x2 DataFrame, saves it to a pickle file, and cleans up afterward.
    """
    test_data = pd.DataFrame([[1, 2], [3, 4]])
    with open(PICKLE_PATH, 'wb') as file:
        pickle.dump(test_data, file)
    yield test_data
    if os.path.exists(PICKLE_PATH):
        os.remove(PICKLE_PATH)

def test_load_data_correctly(pickle_file):
    """
    Test that the `load_periodic_structure_data` function correctly loads data from a pickle file,
    and generates the expected 2D index and column headers with default dimensions.
    """
    df = prep.load_periodic_structure_data(PICKLE_PATH)
    expected_index = ["0.00m", "1.00m"]
    expected_columns = ["0.00m", "1.00m"]

    # Verify data integrity
    pd.testing.assert_frame_equal(df, pickle_file)

    # Verify index and columns
    assert df.index.tolist() == expected_index
    assert df.columns.tolist() == expected_columns

def test_load_data_custom_structure_size(pickle_file):
    """
    Test that the function applies custom structure sizes correctly, producing the expected
    2D index and column headers for specified dimensions.
    """
    df = prep.load_periodic_structure_data(PICKLE_PATH, structure_size=(2, 3))
    expected_index = ["0.00m", "2.00m"]
    expected_columns = ["0.00m", "3.00m"]

    # Verify index and columns with custom structure size
    assert df.index.tolist() == expected_index
    assert df.columns.tolist() == expected_columns

def test_file_not_found_error(caplog):
    """
    Test that the function logs an appropriate error message when the specified file 
    path is invalid.
    """
    prep.load_periodic_structure_data("non_existent_file.pkl")
    # Check that the error message is in the logs
    assert "Error: The file 'non_existent_file.pkl' was not found." in caplog.text

@pytest.fixture(name='empty_pickle_file')
def setup_empty_pickle_file():
    """
    Fixture to set up and tear down an empty DataFrame pickle file for testing.
    This creates an empty DataFrame, saves it to a pickle file, and cleans up afterward.
    """
    empty_data = pd.DataFrame()
    empty_pickle_path = "empty_test_periodic_structure.pkl"
    with open(empty_pickle_path, 'wb') as file:
        pickle.dump(empty_data, file)
    yield empty_pickle_path
    if os.path.exists(empty_pickle_path):
        os.remove(empty_pickle_path)

def test_empty_dataframe_warning(empty_pickle_file, caplog):
    """
    Test that the function logs a warning when an empty DataFrame is loaded from the pickle file.
    """
    df = prep.load_periodic_structure_data(empty_pickle_file)
    assert df.empty
    assert "Warning: DataFrame is empty." in caplog.text
