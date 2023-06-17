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
    


# #################################################################################
# ################## The below functions are copies from other works ##############
# ################### They NEED TO BE EDITED to suit our project ##################
# #################################################################################
#     def get_intensities(self, pair):
#         image = self.load_nifty(pair["image"]).get_fdata().astype(np.float32)
#         label = self.load_nifty(pair["label"]).get_fdata().astype(np.uint8)
#         foreground_idx = np.where(label > 0)
#         intensities = image[foreground_idx].tolist()
#         return intensities
    
#     def collect_intensities(self):
#         intensities = self.run_parallel(self.get_intensities, "training")
#         intensities = list(itertools.chain(*intensities))
#         self.ct_min, self.ct_max = np.percentile(intensities, [0.5, 99.5])
#         self.ct_mean, self.ct_std = np.mean(intensities), np.std(intensities)

# # NB SPACINGS ---> TBD w SEB/Alex --> gives us resolution and resamples if not matching?
#     def load_spacing(image):
#         return image.header["pixdim"][1:4].tolist()[::-1]

#     def get_spacing(self, pair):
#         image = nibabel.load(os.path.join(self.data_path, pair["image"]))
#         spacing = self.load_spacing(image)
#         return spacing
    
#     def collect_spacings(self):
#         spacing = self.run_parallel(self.get_spacing, "training")
#         spacing = np.array(spacing)
#         target_spacing = np.median(spacing, axis=0)
#         if max(target_spacing) / min(target_spacing) >= 3:
#             lowres_axis = np.argmin(target_spacing)
#             target_spacing[lowres_axis] = np.percentile(spacing[:, lowres_axis], 10)
#         self.target_spacing = list(target_spacing)

# ##### THIS IS WM RELATED -- CHECK WITH PEARLY IF ANSIOTROPY IS CALCULABLE FROM T2 OR REQUIRES DTI
#     def check_anisotrophy(self, spacing):
#         def check(spacing):
#             return np.max(spacing) / np.min(spacing) >= 3

#         return check(spacing) or check(self.target_spacing)

#     def resample(self, image, label, image_spacings):
#         if self.target_spacing != image_spacings:
#             image, label = self.resample_pair(image, label, image_spacings)
#         return image, label


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
################ SAMPLE CODE!!!! ##########
# def convert_labels_back_to_BraTS(seg: np.ndarray):
#     new_seg = np.zeros_like(seg)
#     new_seg[seg == 1] = 2
#     new_seg[seg == 3] = 4
#     new_seg[seg == 2] = 1
#     return new_seg


# def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
#     a = sitk.ReadImage(join(input_folder, filename))
#     b = sitk.GetArrayFromImage(a)
#     c = convert_labels_back_to_BraTS(b)
#     d = sitk.GetImageFromArray(c)
#     d.CopyInformation(a)
#     sitk.WriteImage(d, join(output_folder, filename))


# def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
#     """
#     reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
#     """
#     maybe_mkdir_p(output_folder)
#     nii = subfiles(input_folder, suffix='.nii.gz', join=False)
#     with multiprocessing.get_context("spawn").Pool(num_processes) as p:
#         p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))





