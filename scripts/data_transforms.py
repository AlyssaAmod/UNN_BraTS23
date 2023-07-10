from torchvision import transforms
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
        'fakeSSA': tio.OneOf([
            tio.transforms.RandomBlur(std=(0.5, 1.5)),
            tio.transforms.RandomNoise(mean=0, std=(0, 0.33)), # Gaussian noise
            # transforms.ColorJitter(brightness=(0.8, 1.2)),
            tio.transforms.RandomMotion(num_transforms=3, image_interpolation='nearest'),
            tio.transforms.RandomBiasField(coefficients=1),
            tio.transforms.RandomGhosting(intensity=1.5)
        ], p=0.8), # randomly apply ONE of these given transforms with prob 0.5 
        'val': transforms.Compose([
            # transforms.Resize(INPUT_SIZE)
        ]),
        'test' : transforms.Compose([
            # transforms.Resize(INPUT_SIZE)
        ])
    }

    return data_transforms

def transforms_preproc(target_shape=False):
    
    to_ras = tio.ToCanonical() # reorient to RAS+
    # resample_t1space = tio.Resample(image_interpolation='nearest') # target output space (ie. match T2w to the T1w space) 
    if target_shape != False:
        crop_pad = tio.CropOrPad((192, 224, 160))
    else:
        crop_pad = None
    one_hot_enc = tio.OneHot(num_classes=4)
    normalise_foreground = tio.ZNormalization(masking_method=lambda x: x > x.float().mean()) # threshold values above mean only, for binary mask
    # masked = tio.Mask(masking_method=tio.LabelMap(label))
    normalise = tio.ZNormalization()
    fSSA = tio.Compose([
            transforms.RandomRotation((0, 180)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            tio.OneOf([
                tio.transforms.RandomBlur(std=(0.5, 1.5)),
                tio.transforms.RandomNoise(mean=0, std=(0, 0.33)),
                tio.transforms.RandomMotion(num_transforms=3, image_interpolation='nearest'),
                tio.transforms.RandomBiasField(coefficients=1),
                tio.transforms.RandomGhosting(intensity=1.5)], p=0.8)])

        # Define the list of helper functions for the transformation pipeline
    transform_pipeline = {
        'checkRAS' : to_ras,
        'CropOrPad' : crop_pad,
        'ohe' : one_hot_enc,
        'ZnormFore' : normalise_foreground,
        #'MaskNorm' : masked,
        'Znorm': normalise,
        'fSSA' : fSSA}
    
    return transform_pipeline


#### USED IN DATA PREP --> JUST FOR REF HERE
# def apply_preprocessing(subject, transform_pipeline, transL):
#     transformed_subject = subject
#     for transform_name, transform_func in transform_pipeline.items():
#         if transform_name in transL and transform_func is not None:
#             transformed_subject = transform_func(transformed_subject)
#     return transformed_subject

# def load_and_transform_images(data, transform_pipeline, transL):
#     images = []
#     labels = []
#     for item in data:
#         image_path = item["image"]
#         label_path = item["label"]
        
#         # Load the image and label using TorchIO
#         subject = tio.Subject(
#             image=tio.ScalarImage(image_path),
#             label=tio.LabelMap(label_path)
#         )
        
#         # Apply the preprocessing steps
#         transformed_subject = apply_preprocessing(subject, transform_pipeline, transL)
        
#         transformed_image = transformed_subject["image"]
#         transformed_label = transformed_subject["label"]
#         images.append(transformed_image)
#         labels.append(transformed_label)
#     return images, labels

# def preprocess_data(data_dir, args, transList):
#     '''
#     Function that applies all desired preprocessing steps to an image, as well as to its 
#     corresponding ground truth image.

#     Returns: preprocessed image (not yet converted to tensor)
#     img is still a list of arrays of the 4 modalities from data files
#     mask is 3d array
#     return img as list of arrays, and mask as before
#     '''
#     filePaths = json.load(open(data_dir,'dataset.json', "r"))
#     pair = filePaths["training"]

#     outpath = os.path.join(data_dir, args.data_grp + "_prepoc")
#     call(f"mkdir -p {outpath}", shell=True)

#     # transforms = []
#     transform_pipeline = transforms_preproc()
#     # for code, trans in transform_pipeline.items():
#     #     if code in transList:
#     #         transforms.append(trans)
#     # transform = tio.Compose(transforms)
    
#    # Load and transform the images and segmentations
#     transformed_images, transformed_labels = load_and_transform_images(pair, transform_pipeline, transList)

#     # Save the transformed images and segmentations to .npy files
#     for i, (image_path, label_path) in enumerate(zip([item["image"] for item in pair], [item["label"] for item in pair])):
#         image_name = os.path.splitext(os.path.basename(image_path))[0]
#         label_name = os.path.splitext(os.path.basename(label_path))[0]
#         np.save(os.path.join(args.data, image_name[:-4], f"{image_name}.npy"), transformed_images[i].numpy())
#         np.save(os.path.join(args.data, label_name[:-4], f"{label_name}.npy"), transformed_labels[i].numpy())
