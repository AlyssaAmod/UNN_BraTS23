import numpy as np
import torch
import sys


def preprocess_data(img, mask):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
    '''
    # img is still a list of arrays of the 4 modalities from data files
    # mask is 3d array

    # return img as list of arrays, and mask as before
    return img, mask
    

def crop_background(img, mask):
    '''
    Given an image and its mask, crop out unnecessary background pixel (i.e. tightly crop around brain)
    and apply same crop to the ground truth.
    '''

    return img, mask

def convert_subregions():
    '''
    Function to take the given subregion labels and convert them to the subregions needed for challenge evaluation.
    '''