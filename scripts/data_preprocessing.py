import numpy as np
import torch


def preprocess(image, mask):
    '''
    Function that applies all desired preprocessing steps to an image, as well as to its 
    corresponding ground truth image.

    Returns: preprocessed image (not yet converted to tensor)
    '''

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