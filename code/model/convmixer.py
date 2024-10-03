import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import sys

# ConvMixer-256/8 definition
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=10):
    def get_padding(kernel_size):
        pad = kernel_size // 2
        if kernel_size % 2 == 0:
            return (pad - 1, pad)
        else:
            return (pad, pad)

    class DepthwiseConv(nn.Module):
        def __init__(self, dim, kernel_size):
            super().__init__()
            pad_h = get_padding(kernel_size)
            pad_w = get_padding(kernel_size)
            self.conv = nn.Sequential(
                nn.ZeroPad2d((0, 0, pad_h[0], pad_h[1])),
                nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), groups=dim),
                nn.ZeroPad2d((pad_w[0], pad_w[1], 0, 0)),
                nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), groups=dim),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )

        def forward(self, x):
            return self.conv(x)

    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                DepthwiseConv(dim, kernel_size),
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1, 1):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, Residual):
            if hasattr(m.fn, '3'):
                bn = m.fn[3]
                if isinstance(bn, nn.BatchNorm2d):
                    nn.init.zeros_(bn.weight)
                    nn.init.zeros_(bn.bias)

def save_model(model, path='model.pt'):
    """ Save only the model's weights (state_dict) """
    torch.save(model.state_dict(), path)

def train(model, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply Mixup / CutMix
        inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets.argmax(dim=1)).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Epoch [{epoch}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    return test_acc

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Train ConvMixer on CIFAR-10')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--dim', default=128, type=int, help='dimension of ConvMixer')
    parser.add_argument('--depth', default=4, type=int, help='depth of ConvMixer')
    parser.add_argument('--kernel_size', default=8, type=int, help='kernel size of ConvMixer')
    parser.add_argument('--patch_size', default=1, type=int, help='patch size for input')

    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        Cutout(16)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    mixup_fn = timm.data.mixup.Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=10
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvMixer(args.dim, args.depth, kernel_size=args.kernel_size, patch_size=args.patch_size, n_classes=10).to(device)
    initialize_weights(model=model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, scaler)
        test_acc = test(model, test_loader, criterion)
        scheduler.step()

        if test_acc > best_acc:
            print(f'New best accuracy: {test_acc:.2f}%, saving model weights...')
            best_acc = test_acc
            save_model(model, path=f'../../models/model.pt')
