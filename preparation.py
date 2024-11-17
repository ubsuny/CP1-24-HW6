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

def shift_fft(df):
    """
    Shift the 2D FFT data in the DataFrame so that the zero-frequency component is centered.

    Parameters:
    - df: A pandas DataFrame where each cell represents a Fourier-transformed value.

    Returns:
    - A pandas DataFrame with the shifted FFT data.
    """
    # Convert the DataFrame to a numpy array
    fft_data = df.values

    # Perform the shift using np.fft.fftshift to center the zero frequency components
    shifted_fft_data = np.fft.fftshift(fft_data)

    # Convert the shifted data back to a DataFrame with the same index and column headers
    shifted_df = pd.DataFrame(shifted_fft_data, index=df.index, columns=df.columns)

    return shifted_df

def twod_fft_mag(image):
    """this function returns the magnitudes of the frequencies
    (in complex numbers) if you dont want complex numbers, have the returned data and 
    do something simmilar to np.abs(returned data) or np.real
    Also this data is hard to interprate, if you want soemnthing a bit easier to see
    do something like np.log(np.abs(returned data) + 0.001) this makes the data more
    human readible, but once you do this this is no longer correct for the invers fft function
    This function does not cut an image to the correct size
    
    all but 2 lines of this code just makes sure the input data is functioning properly by Inverting
    the calculated fft and compairing it with the original image
    """
    image_array = np.array(image)
    fs = np.fft.fft2(image_array)
    invstimge = np.real(np.fft.ifft2(fs))
    compare = np.isclose(invstimge,image, atol=1e-6)
    alcomp = np.all(compare)
    if not alcomp:
        print("Data is not evenly spaced or data points are missing")
        return None
    return fs

def twod_inv_fft(mag):
    """this returns the original image by using the output of the twod_fft_mag function"""
    newthing = np.fft.ifft2(mag)
    return np.abs(newthing)

def twod_calc_freq(image, width_ofimg, height_ofimg):
    """this takes in the same data as the fft equations only gives the possible frequencies
    of the data in a matrix form, since we are dealing with a 2d image the frequencys will
    come out as a 2d matrix this function should work for a rectangle image aswell as a
    square image but I have not tested this ik a square image works

    width_ofimg, height_ofimg will be in the amount of unit length that the image is
    cut to, so the amount of the meter stick is in the image
    """
    image_array = np.array(image)
    wdth, hght = image_array.shape
    y, x = np.meshgrid(np.fft.fftfreq(wdth), np.fft.fftfreq(hght))
    ynew = y*wdth/width_ofimg
    xnew = x*hght/height_ofimg
    return ynew, xnew

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

def averager(grid):
    """
    averager takes in a two dimensional list of numbers and 
    determines the average absolute value within the 2D list.
    It then returns this average value
    """
    ave=0
    count=0
    for j in grid:
        ave2=0
        count2=0
        for k in j:
            ave2+=np.abs(k)
            count2+=1
        ave+=ave2/(count2)
        count+=1
    ave=ave/(count)

    return ave

def periodic_erase(data):
    """
    periodic erase finds the peaks of the fourier transform
    by assuming that every value above double the average indicates
    periodic behavior. These values above double the average
    are reduced to zero.
    """
    value=data.values
    ave=averager(value)
    new_data=erase(ave*2, value)
    return pd.DataFrame(new_data)

def erase(ave,grid):
    """
    erase takes in a number and a 2D list and
    reduces all values above or equal to ave
    to zero and returns the 2D list.
    """
    new_grid=[]
    row=enumerate(grid)
    for count,j in row:
        new_grid.append([])
        for k in j:
            if np.abs(k)>ave:
                k=0
            new_grid[count].append(k)
    return new_grid
