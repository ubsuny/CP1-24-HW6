'''This module converts an image into a pandas dataframe
then converts the dataframe into a pickle file. The data is '''

import numpy as np
import pandas as pd
from PIL import Image

#Read image data using PIL
img = Image.open('labroom_floor.png')
img_data_array = np.array(img)

#Create dataframe with image data
df = pd.DataFrame({'image_data': [img_data_array]})

#Create pickle file from dataframe
df.to_pickle('img_data.pkl')
