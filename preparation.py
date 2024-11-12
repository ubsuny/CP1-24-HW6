"""Module for image preparation functions."""

import numpy as np

def resize_to_fft_size(image_data):
    """
    Resize the input image (2D array) to the nearest power of two dimensions.
    
    Args:
        image_data (pd.DataFrame): The input image data as a pandas DataFrame.
        
    Returns:
        pd.DataFrame: The resized image data with dimensions as powers of two.
    """
    # Get current image dimensions
    rows, cols = image_data.shape

    # Find the nearest power of two for rows and columns
    new_rows = 2 ** int(np.floor(np.log2(rows)))
    new_cols = 2 ** int(np.floor(np.log2(cols)))

    # Resize the image (either crop or pad the data)
    if new_rows < rows and new_cols < cols:
        resized_image = image_data.iloc[:new_rows, :new_cols]
    else:
        resized_image = image_data.reindex(
            index=range(new_rows), columns=range(new_cols), fill_value=0
        )

    return resized_image
