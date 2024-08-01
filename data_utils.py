import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import log_decorator, load_config

config = load_config()


@log_decorator
def get_data_loaders():
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]

    if config['data_augmentation']:
        transform_list.insert(0, transforms.RandomHorizontalFlip())
        transform_list.insert(1, transforms.RandomRotation(10))

    transform = transforms.Compose(transform_list)

    dataset = datasets.MNIST(root=config['dataset_path'], train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=config['dataset_path'], train=False, download=True, transform=transform)

    train_size = int((1 - config['validation_split']) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    print("Data loaders successfully created.")
    for data_loader in [train_loader, val_loader, test_loader]:
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
