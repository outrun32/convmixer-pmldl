import os
from torchvision import datasets, transforms

def download_cifar10(data_dir='../../data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Define transformations (you can adjust based on your use case)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download the dataset
    print(f"Downloading CIFAR-10 dataset to {data_dir}...")
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    print("Download complete.")

if __name__ == '__main__':
    download_cifar10()
