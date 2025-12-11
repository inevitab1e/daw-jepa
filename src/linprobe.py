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

from src.helper import init_model

logger = getLogger()

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

    name = name.lower()
    if name == "stl10":
        train_t = build_transforms(crop_size, train=True)
        test_t = build_transforms(crop_size, train=False)
        train_set = torchvision.datasets.STL10(
            root=root, split="train", download=True, transform=train_t
        )
        test_set = torchvision.datasets.STL10(
            root=root, split="test", download=True, transform=test_t
        )
        num_classes = 10

    elif name == "cifar10":
        train_t = build_transforms(crop_size, train=True)
        test_t = build_transforms(crop_size, train=False)
        train_set = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_t
        )
        test_set = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_t
        )
        num_classes = 10

    elif name == "cifar100":
        train_t = build_transforms(crop_size, train=True)
        test_t = build_transforms(crop_size, train=False)
        train_set = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=train_t
        )
        test_set = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=test_t
        )
        num_classes = 100

    elif name == "imagenet100":
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


class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


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
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(state["target_encoder"], strict=True)

    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    dummy = torch.randn(2, 3, crop_size, crop_size, device=device)

    H_p, W_p = crop_size // patch_size, crop_size // patch_size
    num_patches = H_p * W_p
    masks_enc = torch.ones((dummy.size(0), 1, num_patches), device=device, dtype=torch.bool)

    feats = encoder(dummy, masks_enc)
    pooled = feats.mean(dim=1)
    feature_dim = pooled.size(-1)

    logger.info(f"Feature dim = {feature_dim}")
    return encoder, feature_dim


@torch.no_grad()
def extract_features(encoder, imgs, patch_size: int):
    device = imgs.device
    B, C, H, W = imgs.shape
    H_p, W_p = H // patch_size, W // patch_size
    num_patches = H_p * W_p
    masks_enc = torch.ones((B, 1, num_patches), device=device, dtype=torch.bool)

    patch_feats = encoder(imgs, masks_enc)
    feats = patch_feats.mean(dim=1)
    return feats


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


def parse_args():
    p = argparse.ArgumentParser("Linear probe on IJEPA ViT encoder")

    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--model_name", type=str, default="vit_small")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--pred_depth", type=int, default=12)
    p.add_argument("--pred_emb_dim", type=int, default=384)

    p.add_argument("--dataset", type=str, required=True,
                   choices=["stl10", "cifar10", "cifar100", "imagenet100"])
    p.add_argument("--data_root", type=str, required=True)

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

    encoder, feat_dim = build_ijepa_encoder(
        device=device,
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        pred_depth=args.pred_depth,
        pred_emb_dim=args.pred_emb_dim,
    )

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
