import os
import argparse
from pathlib import Path

from torchvision.datasets import CIFAR10


def save_split(dataset, split_name, output_root):
    """
    将 CIFAR-10 的一个 split (train/test) 保存为 ImageNet 风格目录：
    output_root/split_name/class_name/xxx.jpg
    """
    split_root = Path(output_root) / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    classes = dataset.classes  # ['airplane', 'automobile', ...]
    for cls in classes:
        (split_root / cls).mkdir(parents=True, exist_ok=True)

    print(f"Saving CIFAR-10 split '{split_name}' with {len(dataset)} images...")

    for idx, (img, label) in enumerate(dataset):
        class_name = classes[label]
        class_dir = split_root / class_name

        filename = f"{class_name}_{idx:06d}.jpg"
        save_path = class_dir / filename

        # CIFAR 返回的 img 是 PIL.Image（在 transform=None 的情况下）
        img.save(save_path, format="JPEG")

        if (idx + 1) % 10000 == 0:
            print(f"  Saved {idx + 1}/{len(dataset)} images for split '{split_name}'")

    print(f"Done saving CIFAR-10 split '{split_name}' to {split_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIFAR-10 dataset to ImageNet-style directory structure."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="CIFAR-10 原始数据下载/存放目录 (torchvision 的 root)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./cifar10_imagenet",
        help="输出的 ImageNet 风格目录根路径",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="不导出 train split",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="不导出 test split (会映射为 val)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    # 训练集 -> train
    if not args.no_train:
        train_set = CIFAR10(
            root=args.data_root,
            train=True,
            download=True,
            transform=None,
        )
        save_split(train_set, "train", args.output_root)

    # 测试集 -> val
    if not args.no_test:
        test_set = CIFAR10(
            root=args.data_root,
            train=False,
            download=True,
            transform=None,
        )
        save_split(test_set, "val", args.output_root)


if __name__ == "__main__":
    main()
# python scripts/cifar10_to_imagenet.py --data-root data/cifar-10-batches-py --output-root data/cifar10_imagenet
