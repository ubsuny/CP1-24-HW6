""" unit test """

import numpy as np
import pandas as pd
from preparation import resize_to_fft_size

def test_resize_to_fft_size():
    # Creates a random 2D pandas DataFrame to simulate an image
    image = np.random.rand(500, 300)
    image_df = pd.DataFrame(image)

    # Resize image
    resized_image = resize_to_fft_size(image_df)

    # Check if the dimensions are powers of two
    assert (resized_image.shape[0] & (resized_image.shape[0] - 1)) == 0  # Rows should be power of 2
    assert (resized_image.shape[1] & (resized_image.shape[1] - 1)) == 0  # Columns should be power of 2
