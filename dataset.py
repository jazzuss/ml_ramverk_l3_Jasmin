import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, test_transform


class CIFAR10Dataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        self.data = datasets.CIFAR10(
            root=root, train=train, download=True,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_transform, test_transform = get_transforms()

    train_set = CIFAR10Dataset(root=data_dir, train=True, transform=train_transform)
    test_set = CIFAR10Dataset(root=data_dir, train=False, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader