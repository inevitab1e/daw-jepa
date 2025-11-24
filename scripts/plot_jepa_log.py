import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def smooth_curve(values, window=5):
    """简单的移动平均平滑，可选"""
    if window <= 1:
        return values
    return np.convolve(values, np.ones(window) / window, mode="same")


def main(csv_path, smooth_window=0):
    print(f"[INFO] Loading csv from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 确保有 epoch 和 loss 两列
    if "epoch" not in df.columns or "loss" not in df.columns:
        raise ValueError(f"CSV must contain 'epoch' and 'loss' columns. Got: {df.columns.tolist()}")

    # 计算每个 epoch 的平均 loss（你也可以改成最后一个 itr 的 loss）
    epoch_avg = df.groupby("epoch")["loss"].mean().reset_index()

    epochs = epoch_avg["epoch"].values
    losses = epoch_avg["loss"].values

    if smooth_window > 1:
        losses_smooth = smooth_curve(losses, window=smooth_window)
    else:
        losses_smooth = losses

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses_smooth, label="avg loss", linewidth=2)
    if smooth_window > 1:
        plt.scatter(epochs, losses, s=10, alpha=0.4, label="raw avg loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss vs epoch from JEPA csv log")
    parser.add_argument("--csv", type=str, required=True, help="Path to csv log file, e.g. jepa_r0.csv")
    parser.add_argument("--smooth", type=int, default=0,
                        help="Moving average window size (0 or 1 means no smoothing, e.g. 5/7)")

    args = parser.parse_args()
    main(args.csv, smooth_window=args.smooth)
