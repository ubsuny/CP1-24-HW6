"""
Module Name: preparation.py

This module provides tools for processing image data and performing frequency analysis using 2D FFT.
The primary use case involves handling image data from a periodic structure, allowing for tasks 
such as windowing, frequency shifting, and removing periodic patterns.

Module Contents:
    - Function to read pickle files with image data, returning a DataFrame with a 2D index based on 
        real-world dimensions.

Functions:
    - load_pickle_data: Reads in pickle files, loads the data and returns a DataFrame with 2D 
    indexing based on real-world dimensions.

"""

import os
import pickle
import numpy as np
import pandas as pd

def dummy():
    """ dummy functions for template file"""
    return 0

def load_pickle_data(pickle_file_path: str, structure_size: tuple = None):
    """
    Loads a pickle file containing a DataFrame, adjusts the index and columns based on
    the dimensions specified (1m x 1m or user-provided size).
    Logs warnings if the DataFrame is empty and handles FileNotFoundError.

    :param pickle_file_path: Path to the pickle file to load.
    :param structure_size: Tuple specifying the custom size for the structure (default is None).
    :return: A pandas DataFrame with appropriate size-based indexing.
    """
    try:
        if not isinstance(pickle_file_path, str) or not os.path.isfile(pickle_file_path):
            err = "The file '{pickle_file_path}' was not found."
            raise ImportError(err)

        with open(pickle_file_path, 'rb') as f:
            df = pickle.load(f)

        if df.empty:
            err = "The loaded DataFrame is empty."
            raise ImportError(err)

        # Adjust index and columns if structure_size is provided
        if structure_size:
            rows, cols = structure_size
            df.index = list(range(0, int(rows)))  # Use integer-based index
            df.columns = list(range(0, int(cols)))  # Use integer-based columns
        else:
            # Default behavior for 1m x 1m structure with integer-based index
            df.index = list(range(0, 2))  # Use integer-based index
            df.columns = list(range(0, 2))  # Use integer-based columns

        return df
    except ImportError as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure
