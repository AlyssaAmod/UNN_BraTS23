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
        'fakeSSA': transforms.Compose([
            # one of?
            # these arent applied with a probability?
            transforms.GaussianBlur(kernel_size=(21, 21), sigma=(0.5, 1.5)),
            tio.transforms.RandomNoise(mean=0, std=(0, 0.33)), # Gaussian noise
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            tio.transforms.RandomMotion(num_transforms=3, image_interpolation='nearest'),
            tio.transforms.RandomBiasField(coefficients=1),
            tio.transforms.RandomGhosting(intensity=1.5)
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