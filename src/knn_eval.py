import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from src.helper import init_model


def build_stl10(root, crop=224):
    transform = T.Compose([
        T.Resize(crop),
        T.CenterCrop(crop),
        T.ToTensor(),
    ])
    train = torchvision.datasets.STL10(root, split="train", download=True, transform=transform)
    test = torchvision.datasets.STL10(root, split="test", download=True, transform=transform)
    return train, test, 10


def build_cifar10(root, crop=224):
    transform = T.Compose([
        T.Resize(crop),
        T.CenterCrop(crop),
        T.ToTensor(),
    ])
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)
    return train, test, 10


@torch.no_grad()
def build_encoder(ckpt_path, device):
    encoder, predictor = init_model(
        device=device,
        patch_size=16,
        crop_size=224,
        pred_depth=12,
        pred_emb_dim=384,
        model_name="vit_small",
    )

    state = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(state["target_encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    def encode(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        num_patches = (H // 16) * (W // 16)
        idx = torch.arange(num_patches, device=x.device).unsqueeze(0).repeat(B, 1)
        patch_feats = encoder(x, idx)
        feats = patch_feats.mean(dim=1)
        return feats

    dummy = torch.randn(2, 3, 224, 224, device=device)
    feat_dim = encode(dummy).shape[-1]

    return encode, feat_dim


@torch.no_grad()
def extract_features(encode, loader, device):
    all_feats = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        feats = encode(imgs)
        all_feats.append(feats)
        all_labels.append(labels)

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


@torch.no_grad()
def knn_classify(
    encode,
    train_loader,
    test_loader,
    device,
    k=20,
    num_classes=10,
    metric="cosine",
):
    print("Extracting train features...")
    train_feats, train_labels = extract_features(encode, train_loader, device)
    n_train, feat_dim = train_feats.shape
    print(f"Train features: {n_train} x {feat_dim}")

    if metric == "cosine":
        train_feats = torch.nn.functional.normalize(train_feats, dim=1)

    print("Evaluating KNN on test set...")
    total = 0
    correct = 0

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        feats = encode(imgs)

        if metric == "cosine":
            feats = torch.nn.functional.normalize(feats, dim=1)
            sim = feats @ train_feats.t()
        else:
            sim = feats @ train_feats.t()

        vals, idx = sim.topk(k=k, dim=1)
        knn_labels = train_labels[idx]

        B = labels.size(0)
        preds = torch.empty(B, dtype=torch.long, device=device)
        for i in range(B):
            counts = torch.bincount(knn_labels[i], minlength=num_classes)
            preds[i] = counts.argmax()

        correct += (preds == labels).sum().item()
        total += B

    acc = 100.0 * correct / total
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", required=True, choices=["stl10", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine"])

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "stl10":
        train_set, test_set, ncls = build_stl10(args.data_root)
    elif args.dataset == "cifar10":
        train_set, test_set, ncls = build_cifar10(args.data_root)
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    encode, feat_dim = build_encoder(args.ckpt_path, device)
    print(f"Feature dim: {feat_dim}")

    acc = knn_classify(
        encode=encode,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        k=args.k,
        num_classes=ncls,
        metric=args.metric,
    )

    print(f"[KNN] Dataset: {args.dataset}, k={args.k}, acc = {acc:.2f}")


if __name__ == "__main__":
    main()
