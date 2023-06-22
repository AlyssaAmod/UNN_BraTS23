""" This script is to prepare the provided data set for pre-processing.

It will take as input a directory path to the training data and apply the following:

1. Read in the dataset folder structure
2. Store variables
    subjID = subject IDs
    img_dir = path to each imaging modality
    lbl dir = path to corresponding segmentation mask
3. Load nifty file for each modality
4. Extract voxel intensity values, header information and affine matrix
    Stack voxel data from each modality into 1 tensor
    cropping out background
    Add an extra channel for one hot encoding??
5. Save the following files:
    images/subjIDxxx.nii.gz = stacked modalities output into a nifti file in an images folder
    labels/subjIDxxx_seg.nii.gz = segmentation mask
6. Create and save json file that contains a dictionary of dictionaries and lists:
    A dictionary of dummy coding for seg labels as provided by BraTS
        "labels" : {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
    A dictionary of dummy coding for each modality
        "modality": {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
    A dictionary of dictionaries containing the image-label path pairs
        "training": [{"image": "images/subjIDxxx.nii.gz", "label": "labels/subjIDxxx_seg.nii.gz"}

Add noise defs for fake SSA data in an if 
"""