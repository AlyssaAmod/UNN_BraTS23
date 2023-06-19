from torchvision import transforms
import torchio as tio

#! TO DO: we must fill in the transforms we want to apply
def define_transforms():
    # Initialise data transforms
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(INPUT_SIZE),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
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