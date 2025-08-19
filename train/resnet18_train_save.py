import os, time, json
from pathlib import Path
import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.resnet import resnet18

# config (실험 설정)
cfg = {
    "batch_size": 20,
    "lr": 0.003,
    "epochs": 100,
    "optimizer": "SGD(momentum=0.9, weight_decay=5e-4)",
    "architecture": "resnet18",
    "dataset": "CIFAR-10",
    "image_size": 224,
    "normalize_mean": (0.5, 0.5, 0.5),
    "normalize_std": (0.5, 0.5, 0.5),
    "seed": 42
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# routes
SAVE_ROOT = Path("./runs")
RUN_DIR = SAVE_ROOT / f"resnet18_{time.strftime('%Y%m%d_%H%M%S')}"
(RUN_DIR/"checkpoints").mkdir(parents=True, exist_ok=True)
(RUN_DIR/"exports").mkdir(parents=True, exist_ok=True)

# save config.json
with open(RUN_DIR/"config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

RESUME = ""

# datasets
datapath = './data'
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop((cfg["image_size"], cfg["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"])
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((cfg["image_size"], cfg["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"])
])

train_dataset = datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform_train)
test_dataset  = datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

# model, criterion, optimizer
model = resnet18(num_classes=10).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=5e-4)

start_epoch = 0
best_acc = 0.0

# load checkpoint
def load_checkpoint(path):
    global start_epoch, best_acc, cfg
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    meta = ckpt.get("meta", {})
    start_epoch = meta.get("epoch",0)+1
    best_acc = meta.get("best_metric", 0.0)
    cfg = meta.get("config", cfg)
    print(f"[resume] loaded epoch = {start_epoch-1}, best_acc = {best_acc:.4f}")

# save checkpoint
def save_checkpoint(tag, epoch, best_metric):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": {"epoch": epoch, "best_metric": best_metric},
        "config": cfg
    }
    path = RUN_DIR/"checkpoints"/f"{tag}.pt"
    torch.save(ckpt, path)
    return str(path)

# load resume
if RESUME:
    load_checkpoint(RESUME)

print("save dir:", RUN_DIR)

val_acc = best_acc

for epoch in range(cfg["epochs"]):
    # train one epoch
    model.train()
    total_loss = 0.0
    correct = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False)
    for x,y in loop:
        x,y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(y).sum().item()

        loop.set_postfix(loss = loss.item())
    
    train_acc = correct / len(train_loader.dataset)
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")

    # validate
    model.eval()
    correct = 0
    previous_val_acc = val_acc
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = outputs.max(dim=1)
            correct += (preds == y).sum().item()
        val_acc = correct/len(test_loader.dataset)
    print(f"validation Accuracy: {val_acc:.4f}")

    # scheduling
    if val_acc <= previous_val_acc:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
    
    # save checkpoints
    last_path = save_checkpoint("last", epoch, best_metric=max(best_acc, val_acc))
    if val_acc > best_acc:
        best_acc = val_acc
        best_path = save_checkpoint("best", epoch, best_metric=best_acc)
    
    # state_dict only
    torch.save(model.state_dict(), RUN_DIR/"checkpoints"/"deploy_weights.pt")

# test
model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, preds = outputs.max(dim=1)
        correct += (preds == y).sum().item()
final_acc = correct / len(test_loader.dataset)
print(f"Test Accuracy: {final_acc:.4f}")
print(f"Artifacts: {RUN_DIR}")
