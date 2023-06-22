from torchvision import transforms
import torchio as tio

#! TO DO: we must fill in the transforms we want to apply
def define_transforms():
    # Initialise data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((0, 180)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3)
        ]),
        'trainSSA': transforms.Compose([
            transforms.GaussianBlur(kernel_size=(21, 21), sigma=(0.5, 1.5)),
            tio.transforms.RandomNoise(mean=0, std=(0, 0.33)), # Gaussian noise
            transforms.ColorJitter(brightness=(0.8, 1.2))
        ]),
        'val': transforms.Compose([
            # transforms.Resize(INPUT_SIZE),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
        ]),
        'test' : transforms.Compose([
            # transforms.Resize(INPUT_SIZE),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
        ])
    }

    return data_transforms