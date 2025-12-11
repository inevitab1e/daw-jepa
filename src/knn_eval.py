import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from src.helper import init_model


# ---- Dataset ----

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


# ---- Encoder wrapper ----

@torch.no_grad()
def build_encoder(ckpt_path, device):
    # 跟你 linear probe 一样：用 target_encoder 当 frozen encoder
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
        """
        x: [B, 3, 224, 224]
        return: [B, D] global feature
        """
        B, C, H, W = x.shape
        num_patches = (H // 16) * (W // 16)  # e.g. 14x14=196
        idx = torch.arange(num_patches, device=x.device).unsqueeze(0).repeat(B, 1)
        patch_feats = encoder(x, idx)        # [B, L, D]
        feats = patch_feats.mean(dim=1)      # [B, D]
        return feats

    # 推一下维度
    dummy = torch.randn(2, 3, 224, 224, device=device)
    feat_dim = encode(dummy).shape[-1]

    return encode, feat_dim


# ---- KNN 逻辑 ----

@torch.no_grad()
def extract_features(encode, loader, device):
    """
    把一个 dataloader 里的所有图片提成特征 & label。
    返回：
      feats:  [N, D] (device 上)
      labels: [N]   (device 上, long)
    """
    all_feats = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        feats = encode(imgs)  # [B, D]
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
    """
    标准 KNN：
      - 先提取 train features & labels
      - 对每个 test batch：
          * 提特征
          * 和所有 train feature 算相似度
          * 取 top-k 做投票
    """
    # 1. 提取 train 全部特征
    print("Extracting train features...")
    train_feats, train_labels = extract_features(encode, train_loader, device)
    n_train, feat_dim = train_feats.shape
    print(f"Train features: {n_train} x {feat_dim}")

    # 2. 归一化特征（cosine 用）
    if metric == "cosine":
        train_feats = torch.nn.functional.normalize(train_feats, dim=1)

    # 3. 遍历 test
    print("Evaluating KNN on test set...")
    total = 0
    correct = 0

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        feats = encode(imgs)  # [B, D]

        if metric == "cosine":
            feats = torch.nn.functional.normalize(feats, dim=1)
            # 相似度 = 余弦相似度
            # [B, D] x [D, N_train] -> [B, N_train]
            sim = feats @ train_feats.t()
        else:
            # 简单 L2 距离：||x - y||^2 = x^2 + y^2 - 2 x·y
            # 这里先算 inner product
            sim = feats @ train_feats.t()
            # 再构造距离：我们可以直接用 sim 排序，但为了清晰可以转换成负距离
            # sim 越大 → 距离越近，下面统一用 topk 即可
        # 取 top-k 最近邻
        # sim: [B, N_train]
        vals, idx = sim.topk(k=k, dim=1)  # idx: [B, k]

        # 取邻居的标签
        knn_labels = train_labels[idx]     # [B, k]

        # 多数投票（可以无权，也可以用 vals 做权重，这里先简单无权）
        # B 不大，可以用 for-loop
        B = labels.size(0)
        preds = torch.empty(B, dtype=torch.long, device=device)
        for i in range(B):
            # knn_labels[i]: [k]
            counts = torch.bincount(knn_labels[i], minlength=num_classes)
            preds[i] = counts.argmax()

        correct += (preds == labels).sum().item()
        total += B

    acc = 100.0 * correct / total
    return acc


# ---- Main ----

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

    # build dataset
    if args.dataset == "stl10":
        train_set, test_set, ncls = build_stl10(args.data_root)
    elif args.dataset == "cifar10":
        train_set, test_set, ncls = build_cifar10(args.data_root)
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # build encoder
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


# python -m src.knn_eval \
#   --ckpt_path logs/jepa-latest.pth.tar \
#   --data_root data \
#   --dataset stl10 \
#   --k 20 \
#   --batch_size 256
