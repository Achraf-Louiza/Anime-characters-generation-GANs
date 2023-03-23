import zipfile
from PIL import Image
import io
import numpy as np
from tqdm import tqdm 

def read_data(zip_path):
    images = []
    fixed_shape = (40, 40)
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get a list of all files in the zip file
        image_files = [f for f in zip_ref.namelist() if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
        
        # Iterate over each image file
        for image_file in tqdm(image_files):
            # Read the image data from the zip file
            with zip_ref.open(image_file) as file:
                image_data = file.read()
            # Open the image using Pillow
            image = Image.open(io.BytesIO(image_data))
            image = np.array(image.resize(fixed_shape))
            images.append(image)
    images = np.asarray(images)
    return images