from torchvision import transforms
import albumentations as A
import torchio as tio
import torch
import warnings


#! TO DO: we must fill in the transforms we want to apply
def define_transforms():
    # Initialise data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((0, 180)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3)
        ]),
        'fakeSSA': A.OneOf([
            transforms.GaussianBlur(kernel_size=(21, 21), sigma=(0.5, 1.5)),
            tio.transforms.RandomNoise(mean=0, std=(0, 0.33)), # Gaussian noise
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            tio.transforms.RandomMotion(num_transforms=3, image_interpolation='nearest'),
            tio.transforms.RandomBiasField(coefficients=1),
            tio.transforms.RandomGhosting(intensity=1.5)
        ], p=0.5), # randomly apply ONE of these given transforms with prob 0.5 
        'val': transforms.Compose([
            # transforms.Resize(INPUT_SIZE)
        ]),
        'test' : transforms.Compose([
            # transforms.Resize(INPUT_SIZE)
        ])
    }

    return data_transforms

def transforms_preproc(image, ohe, target_shape):
    
    to_ras = tio.ToCanonical() # reorient to RAS+
    resample_t1space = tio.Resample(image_interpolation='nearest') # target output space (ie. match T2w to the T1w space) 
    if target_shape != None:
        crop_pad = tio.CropOrPad(target_shape)
    one_hot_enc = tio.OneHot(num_classes=4)
    normalise_foreground = tio.ZNormalization(masking_method=lambda x: x > x.float().mean()) # threshold values above mean only, for binary mask
    masked = tio.Mask(masking_method=tio.LabelMap(label))
    normalise = tio.ZNormalization()

        
    apply_trans = {
        'checkRAS' : to_ras,
        'resampleTot1' : resample_t1space,
        'crop_pad' : crop_pad,
        'oheZN' : tio.Compose([one_hot_enc, normalise_foreground]),
        'brainmask' : tio.Compose([ masked, normalise])
    }

        # Define the list of helper functions for the transformation pipeline
    transform_pipeline = [
        to_ras,
        crop_pad,
        one_hot_enc,
        normalise_foreground,
        masked,
        normalise
    ]
    
    return apply_trans, transform_pipeline

def apply_transforms(image, transforms, seed=42, show=False, exclude=None):
    torch.manual_seed(seed)
    results = []
    transformed = image
    for transform in transforms:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform(transformed)
            if exclude is None or transform.name not in exclude:
                transformed = result

