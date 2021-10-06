import os
import numpy as np
from tqdm import tqdm
from pydicom import dcmread

def decode(dir_path: str):
    # parameter check
    assert os.path.isdir(dir_path), 'no {} or not a directory'.format(dir_path)
    dcm_list = os.listdir(dir_path)
    assert len(dcm_list) > 0, 'no image file in given directory'
    
    # sort dcm filenames
    dcm_list.sort()

    # get width, height from the one of dcm files
    dcm_path = os.path.join(dir_path, dcm_list[0])
    shape = dcmread(dcm_path).pixel_array.shape

    # decode with shape consistency check
    image_list = []
    for dcm_filename in tqdm(dcm_list):
        dcm_path = os.path.join(dir_path, dcm_filename)
        image = dcmread(dcm_path).pixel_array
        image = np.maximum(image, 0)
        
        assert image.shape == shape, 'image size not consistent'
        
        image_list.append(image)
    
    # return np.ndarray[D][H][W], np.uint16, -1000 ~ 1000
    image_tensor = np.array(image_list)
    return image_tensor