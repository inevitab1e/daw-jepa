import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from src.helper import init_model


class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


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
    # Build JEPA encoder (teacher)
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

    # Wrapper: encoder(x) -> global feature
    def encode(x):
        B, C, H, W = x.shape
        num_patches = (H // 16) * (W // 16)      # 196
        # correct mask format: [B, L], int indices
        idx = torch.arange(num_patches, device=x.device).unsqueeze(0).repeat(B, 1)
        # forward
        patch_feats = encoder(x, idx)   # [B, L, D]
        # global feature via mean-pool
        return patch_feats.mean(dim=1)  # [B, D]

    # Infer feature dimension
    dummy = torch.randn(2, 3, 224, 224, device=device)
    feat_dim = encode(dummy).shape[-1]

    return encode, feat_dim


# ---- Train & Test ----

def train_one_epoch(encode, head, loader, opt, device):
    head.train()
    ce = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            feats = encode(imgs)

        logits = head(feats)
        loss = ce(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, 100 * correct / total


@torch.no_grad()
def evaluate(encode, head, loader, device):
    head.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        feats = encode(imgs)
        logits = head(feats)
        loss = ce(logits, labels)

        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, 100 * correct / total


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "stl10":
        train_set, test_set, ncls = build_stl10(args.data_root)
    elif args.dataset == "cifar10":
        train_set, test_set, ncls = build_cifar10(args.data_root)
    else:
        raise ValueError("Unsupported dataset")

    # train_set, test_set, ncls = build_stl10(args.data_root)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    encode, feat_dim = build_encoder(args.ckpt_path, device)
    head = LinearProbe(feat_dim, ncls).to(device)
    opt = optim.SGD(head.parameters(), lr=args.lr, momentum=0.9)
    # opt = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0
    for ep in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(encode, head, train_loader, opt, device)
        te_loss, te_acc = evaluate(encode, head, test_loader, device)
        # scheduler.step()
        best_acc = max(best_acc, te_acc)

        print(f"Epoch {ep+1}: Train {tr_acc:.2f}  Test {te_acc:.2f}  Best {best_acc:.2f}")


if __name__ == "__main__":
    main()
