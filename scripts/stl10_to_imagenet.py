import os
import argparse
from pathlib import Path

from torchvision.datasets import STL10


# STL10 的 10 个类别，对应官方顺序
STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def save_split(dataset, split_name, output_root):
    """
    将一个 split (train/test) 的数据集保存为 ImageNet 风格目录：
    output_root/split_name/class_name/xxx.jpg
    """
    split_root = Path(output_root) / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    # 先把每个类别的文件夹建好
    for cls in STL10_CLASSES:
        (split_root / cls).mkdir(parents=True, exist_ok=True)

    print(f"Saving split '{split_name}' with {len(dataset)} images...")

    for idx, (img, label) in enumerate(dataset):
        class_name = STL10_CLASSES[label]
        class_dir = split_root / class_name

        # 文件名：class_idx_xxxxxx.jpg
        filename = f"{class_name}_{idx:06d}.jpg"
        save_path = class_dir / filename

        # dataset 返回的 img 本身就是 PIL.Image
        img.save(save_path, format="JPEG")

        if (idx + 1) % 1000 == 0:
            print(f"  Saved {idx + 1}/{len(dataset)} images for split '{split_name}'")

    print(f"Done saving split '{split_name}' to {split_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert STL10 dataset to ImageNet-style directory structure."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="STL10 原始数据下载/存放目录 (torchvision 的 root)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./stl10_imagenet",
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

    # 导出训练集 -> ImageNet 的 train
    if not args.no_train:
        train_set = STL10(
            root=args.data_root,
            split="train",
            download=True,
            transform=None,  # 返回 PIL.Image
        )
        save_split(train_set, "train", args.output_root)

    # 导出测试集 -> ImageNet 的 val
    if not args.no_test:
        test_set = STL10(
            root=args.data_root,
            split="test",
            download=True,
            transform=None,
        )
        # 为了和 ImageNet 对齐，这里用 "val" 作为目录名
        save_split(test_set, "val", args.output_root)


if __name__ == "__main__":
    main()
# python scripts/stl10_to_imagenet.py --data-root data/stl10 --output-root data/stl10_imagenet

# python main.py --fname configs/slt10_vits16_ep100.yaml --devices cuda:0