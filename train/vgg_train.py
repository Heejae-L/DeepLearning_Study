import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from torchvision import utils
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from model.vgg import VGG


#하이퍼파라미터
batch_size = 20
lr = 0.003
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#데이터셋 로드
datapath = './data'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_dataset = datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델
model = VGG(10).to(device)

# 손실 함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)

val_acc = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()

        loop.set_postfix(loss=loss.item())

    train_acc = correct / len(train_loader.dataset)
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")

    model.eval()
    correct = 0
    previouse_val_acc = val_acc
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
    val_acc = correct / len(test_loader.dataset)

    print(f"Validation Accuracy: {val_acc:.4f}")
    
    if val_acc <= previouse_val_acc:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1


model.eval()
correct = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
val_acc = correct / len(test_loader.dataset)

print(f"Test Accuracy: {val_acc:.4f}")