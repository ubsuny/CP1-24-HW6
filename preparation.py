""" dummy preparations file """
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
