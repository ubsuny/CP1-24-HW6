""" This Module contain function to shuft FFT data """

import numpy as np
import pandas as pd

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
