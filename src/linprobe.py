# src/linprobe.py
# Linear probe on top of a frozen IJEPA ViT encoder.

import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

from logging import getLogger

# 复用 IJEPA 里的工具
from src.helper import init_model  # 用来构建 ViT encoder/predictor

logger = getLogger()


# -------------------------
# 数据集构建
# -------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(crop_size: int, train: bool = True):
    if train:
        transform = T.Compose([
            T.Resize(crop_size),
            T.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        transform = T.Compose([
            T.Resize(crop_size + 32),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transform


def build_dataset(
    name: str,
    root: str,
    crop_size: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    """
    支持 stl10 / cifar10 / cifar100 / imagenet100 (ImageFolder 格式)
    返回 train_set, test_set, num_classes
    """
    name = name.lower()
    if name == "stl10":
        train_t = build_transforms(crop_size, train=True)
        test_t = build_transforms(crop_size, train=False)
        train_set = torchvision.datasets.STL10(
            root=root, split="train", download=False, transform=train_t
        )
        test_set = torchvision.datasets.STL10(
            root=root, split="test", download=False, transform=test_t
        )
        num_classes = 10

    elif name == "cifar10":
        train_t = build_transforms(crop_size, train=True)
        test_t = build_transforms(crop_size, train=False)
        train_set = torchvision.datasets.CIFAR10(
            root=root, train=True, download=False, transform=train_t
        )
        test_set = torchvision.datasets.CIFAR10(
            root=root, train=False, download=False, transform=test_t
        )
        num_classes = 10

    elif name == "cifar100":
        train_t = build_transforms(crop_size, train=True)
        test_t = build_transforms(crop_size, train=False)
        train_set = torchvision.datasets.CIFAR100(
            root=root, train=True, download=False, transform=train_t
        )
        test_set = torchvision.datasets.CIFAR100(
            root=root, train=False, download=False, transform=test_t
        )
        num_classes = 100

    elif name == "imagenet100":
        # 假设你已经把 ImageNet100 转成标准 ImageFolder 结构：
        #   root/train/<class_x>/*.jpeg
        #   root/val/<class_x>/*.jpeg
        train_t = build_transforms(crop_size, train=True)
        test_t = build_transforms(crop_size, train=False)
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_t)
        test_set = torchvision.datasets.ImageFolder(val_dir, transform=test_t)
        num_classes = len(train_set.classes)

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_set, test_set, num_classes


# -------------------------
# 线性 probe 模型
# -------------------------

class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# -------------------------
# IJEPA encoder 封装
# -------------------------

@torch.no_grad()
def build_ijepa_encoder(
    device,
    ckpt_path: str,
    model_name: str = "vit_small",
    patch_size: int = 16,
    crop_size: int = 224,
    pred_depth: int = 12,
    pred_emb_dim: int = 384,
):
    """
    使用 IJEPA 的 init_model 构建 encoder，并从 ckpt 加载权重。
    返回：
      - encoder (已冻结, eval 模式)
      - feature_dim (线性 probe 输入维度)
    """
    logger.info("Building IJEPA encoder...")
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )

    # 加载预训练参数
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(state["encoder"], strict=True)

    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 用一个 dummy batch 推一下，推断 feature_dim
    dummy = torch.randn(2, 3, crop_size, crop_size, device=device)

    # 构建「全 context mask」，所有 patch 都在 context 里
    H_p, W_p = crop_size // patch_size, crop_size // patch_size
    num_patches = H_p * W_p
    full_idx = torch.arange(num_patches, dtype=torch.int32, device=device)
    # 形状 [B, nenc, L_enc]，这里 nenc=1
    masks_enc = full_idx.unsqueeze(0).unsqueeze(0).repeat(dummy.size(0), 1, 1)

    # encoder(imgs, masks_enc) 返回 [B, L_ctx, D] 的 patch 表征
    feats = encoder(dummy, masks_enc)   # 你现有的 encoder 接口就是这样用的
    # 我们对 patch 维度做 mean-pool 得到 [B, D]
    pooled = feats.mean(dim=1)
    feature_dim = pooled.size(-1)

    logger.info(f"Feature dim = {feature_dim}")
    return encoder, feature_dim


@torch.no_grad()
def extract_features(encoder, imgs, patch_size: int):
    """
    给一批图片 imgs，返回 [B, D] 的图像级特征。
    做法：
      - 构建全 context mask（所有 patch 都视为 context）
      - 调用 encoder(imgs, masks_enc)
      - 对 patch 做 mean-pool
    """
    device = imgs.device
    B, C, H, W = imgs.shape
    H_p, W_p = H // patch_size, W // patch_size
    num_patches = H_p * W_p
    full_idx = torch.arange(num_patches, dtype=torch.int32, device=device)
    masks_enc = full_idx.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)  # [B, 1, L]

    patch_feats = encoder(imgs, masks_enc)  # [B, L_ctx, D]
    feats = patch_feats.mean(dim=1)         # [B, D]
    return feats


# -------------------------
# 训练 & 验证
# -------------------------

def train_one_epoch(
    encoder,
    linear_head,
    dataloader,
    optimizer,
    device,
    patch_size: int,
):
    encoder.eval()
    linear_head.train()

    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            feats = extract_features(encoder, imgs, patch_size)

        logits = linear_head(feats)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = logits.max(1)
        total_correct += (preds == targets).sum().item()
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples * 100.0
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    encoder,
    linear_head,
    dataloader,
    device,
    patch_size: int,
):
    encoder.eval()
    linear_head.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        feats = extract_features(encoder, imgs, patch_size)
        logits = linear_head(feats)
        loss = criterion(logits, targets)

        total_loss += loss.item() * imgs.size(0)
        _, preds = logits.max(1)
        total_correct += (preds == targets).sum().item()
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples * 100.0
    return avg_loss, acc


# -------------------------
# main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser("Linear probe on IJEPA ViT encoder")

    # 模型 & ckpt
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to your pre-trained IJEPA checkpoint (.pth.tar)")
    p.add_argument("--model_name", type=str, default="vit_small",
                   help="IJEPA model_name, e.g., vit_small / vit_base / ...")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--pred_depth", type=int, default=12)
    p.add_argument("--pred_emb_dim", type=int, default=384)

    # 数据
    p.add_argument("--dataset", type=str, required=True,
                   choices=["stl10", "cifar10", "cifar100", "imagenet100"])
    p.add_argument("--data_root", type=str, required=True,
                   help="Root path for the chosen dataset")

    # 训练超参
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 构建数据集 & dataloader
    train_set, test_set, num_classes = build_dataset(
        name=args.dataset,
        root=args.data_root,
        crop_size=args.crop_size,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2. 构建 IJEPA encoder（冻结）
    encoder, feat_dim = build_ijepa_encoder(
        device=device,
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        pred_depth=args.pred_depth,
        pred_emb_dim=args.pred_emb_dim,
    )

    # 3. 线性头
    linear_head = LinearProbe(feat_dim, num_classes).to(device)

    optimizer = optim.SGD(
        linear_head.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            encoder, linear_head, train_loader, optimizer, device, patch_size=args.patch_size
        )
        test_loss, test_acc = evaluate(
            encoder, linear_head, test_loader, device, patch_size=args.patch_size
        )

        if test_acc > best_acc:
            best_acc = test_acc

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} "
            f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f} "
            f"(Best: {best_acc:.2f})"
        )


if __name__ == "__main__":
    main()
