import ssl
import os
from PIL import Image

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import random_split, DataLoader, Dataset

from config import IMAGE_SIZE, TRAIN_RATIO, BATCH_SIZE



class CustomPetDataset(Dataset):
    """A customized dataset pets based on species.
    
    Oxford-IIIT Pets dataset is based on breeds.
    This class, convenrts that dataset to a species-based
    one, using `annotations/list.txt` file.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, "images")
        list_txt_path = os.path.join(root, "annotations", "list.txt")
        self.samples = []

        with open(list_txt_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    # do not consider headers
                    continue
                parts = line.strip().split()
                filename = parts[0] + ".jpg"
                species = int(parts[2])  # 1=dog, 2=cat
                label = 1 if species == 1 else 0  # 1=dog, 0=cat
                self.samples.append((filename, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TransformedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)


def get_breeds_dataloader() -> tuple[DataLoader, DataLoader]:
    """Download, split and load dataset.

    Downloads Oxford-IIIT Pet dataset if it doesn't exist.
    Only downloads categories, not segmentations.
    Splits dataset into training and testing sets.
    Loads them into DataLoaders.
    Lables are animal breeds.

    Returns:
        tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) # mean and standard deviation for each color channel
        #  these are numbers used in training model one ImageNet problem
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) # mean and standard deviation for each color channel
        #  these are numbers used in training model one ImageNet problem
    ])

    ssl._create_default_https_context = ssl._create_unverified_context
    dataset = OxfordIIITPet(
        root="data/",
        download=True,
        target_types="category"
    )

    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataset, test_dataset = TransformedDataset(train_dataset, train_transform), TransformedDataset(test_dataset, test_transform)

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dl, test_dl


def get_species_dataloader() -> tuple[DataLoader, DataLoader]:
    """Download, split and load dataset.

    Downloads Oxford-IIIT Pet dataset if it doesn't exist.
    Only downloads categories, not segmentations.
    Splits dataset into training and testing sets.
    Loads them into DataLoaders.
    Lables are animal species (cat vs. dog only).

    Returns:
        tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    """
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ) # mean and standard deviation for each color channel
    ])

    ssl._create_default_https_context = ssl._create_unverified_context

    dataset = CustomPetDataset(root="data/oxford-iiit-pet", transform=transform)

    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dl, test_dl
