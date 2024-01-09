import os
import PIL.Image
from random import randrange
import numpy as np

def load_kodak(img_size: int)-> np.ndarray:
    """
    Load the Kodak dataset with specified size.
    """
    img_path = os.path.join(os.path.dirname(__file__), 'kodak_dataset')
    
    #get list of images
    img_list = []
    for filename in os.listdir(img_path):
        img = PIL.Image.open(os.path.join(img_path, filename))
        #crop image
        x,y = img.size
        x1 = randrange(0, x - img_size)
        y1 = randrange(0, y - img_size)
        img = img.crop((x1, y1, x1 + img_size, y1 + img_size))
        img_list.append(np.array(img))

    return np.array(img_list)

def load_CIFAR10(batch_size: int):
    #load CIFAR10
    import torch
    import torchvision
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [transforms.ToTensor(),
        Rescale()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return trainloader, testloader

class Rescale(object):
    def __call__(self,sample):
        image = sample/255.
        return image
    
def load_ImageNet(size, batch_size: int, path: str, num_workers: int = 1):
    import torch
    import torchvision
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(size, scale=(0.999, 1.0)),
        transforms.ToTensor()])
    
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'val')

    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, test_dataloader