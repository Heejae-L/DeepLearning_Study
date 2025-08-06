import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.densenet import DenseNet121

# 하이퍼파라미터
batch_size = 20
epochs = 30
learning_rate = 0.003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# 데이터 전처리 및 로딩
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 모델 생성
model = DenseNet121(num_classes=10).to(device)

# 손실함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()

        loop.set_postfix(loss=loss.item())

    train_acc = correct / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")

    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = outputs.max(dim = 1)
            correct += (preds == y).sum().item()
    val_acc = correct / len(test_loader.dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")

# 테스트
model.eval()
correct = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, preds = outputs.max(dim = 1)
        correct += (preds == y).sum().item()
test_acc = correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_acc:.4f}")
