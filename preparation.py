"""
Module Name: preparation.py

This module provides tools for processing image data and performing frequency analysis using 2D FFT.
The primary use case involves handling image data from a periodic structure, allowing for tasks 
such as windowing, frequency shifting, and removing periodic patterns.

Module Contents:
    - pytest fixtures for unit tests of input data for FFT and pandas DataFrames.
    - Function to adjust image dimensions to be FFT-compatible.
    - Function to read pickle files with image data, returning a DataFrame with a 2D index based on 
        real-world dimensions.
    - FFT-related functions, including shifting data, computing 2D FFT and inverse FFT, and 
        calculating frequencies.
    - Functions for 2D windowing/unwindowing and removing periodic patterns.

Functions:
    - fixture_input_dataframe: Pytest fixture that provides a sample input DataFrame for unit tests.
    - fixture_fft_data: Pytest fixture for generating synthetic FFT data for testing purposes.
    - adjust_image_for_fft: Adjusts the dimensions of an image for compatibility with FFT by 
    cropping or padding.
    - load_periodic_structure_data: Reads in pickle files, returns a DataFrame with 2D indexing 
    based on real-world dimensions.
    - shift_fft_data: Shifts FFT data to center the low frequencies.
    - perform_2d_fft: Applies 2D FFT and inverse FFT, calculating frequencies in useful units.
    - apply_2d_window: Applies windowing/unwindowing functions to reduce spectral leakage.
    - remove_periodic_pattern: Removes a periodic pattern from the data.

"""

import numpy as np
import pandas as pd

def load_periodic_structure_data(pickle_path: str, structure_size=(1, 1)):
    """
    Loads data from a pickle file generated from a Blender scene with a periodic structure
    and returns it as a pandas DataFrame with a 2D index based on the specified dimensions.

    Parameters:
    - pickle_path: str - The file path to the pickle file.
    - structure_size: tuple - The (width, height) dimensions of the periodic structure, 
    defaulting to (1m, 1m).

    Returns:
    - pd.DataFrame - A DataFrame with a 2D index/column header based on actual object dimensions.
    """
    try:
        # Load the pickle file into a DataFrame
        df = pd.read_pickle(pickle_path)

        # Check the structure of the DataFrame
        if df.empty:
            print("Warning: DataFrame is empty.")
            return df

        # Extract the number of rows and columns from the DataFrame
        num_rows, num_cols = df.shape

        # Generate index and columns based on structure size
        row_labels = [f"{i * structure_size[0]:.2f}m" for i in range(num_rows)]
        col_labels = [f"{j * structure_size[1]:.2f}m" for j in range(num_cols)]

        # Assign the generated labels to the DataFrame
        df.index = row_labels
        df.columns = col_labels

        return df

    except FileNotFoundError:
        print(f"Error: The file '{pickle_path}' was not found.")
        return None
    except ImportError as e:
        print(f"Unable to import pickle file data {e}")
        return None

# Example usage:
# df = load_periodic_structure_data("path/to/pickle/file.pkl")
