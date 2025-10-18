import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from torch.utils.data import Subset
from src.logger import get_logger
from src.exception import AppException

logger = get_logger("data")


def is_image_file(filename):
    try:
        Image.open(filename).verify()
        return True
    except Exception:
        return False


class SafeImageFolder(ImageFolder):
    """ImageFolder that filters out unreadable files."""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.samples = [s for s in self.samples if is_image_file(s[0])]
        self.targets = [s[1] for s in self.samples]


def get_transforms(input_size=224):
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(10, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def create_dataloaders(data_path, train_transforms, val_transforms, batch_size=64, num_workers=2):
    if not os.path.exists(data_path):
        logger.error(f"Dataset path not found: {data_path}")
        raise AppException(f"Dataset path not found: {data_path}")

    full_dataset = SafeImageFolder(data_path, transform=train_transforms)
    filenames = np.array([os.path.basename(path[0]) for path in full_dataset.samples])
    labels = np.array([path[1] for path in full_dataset.samples])
    groups = np.array([f.split('_')[1] if '_' in f else f for f in filenames])

    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(filenames, labels, groups=groups))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(SafeImageFolder(data_path, transform=val_transforms), val_idx)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    logger.info(f"Created dataloaders. Classes: {full_dataset.classes}")
    return train_loader, val_loader, full_dataset.classes
