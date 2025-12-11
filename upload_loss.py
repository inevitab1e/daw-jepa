import wandb
import pandas as pd

# 🔧 your CSV files
csv_files = {
    "baseline": "/Users/lucas/Library/CloudStorage/GoogleDrive-jwu14572@gmail.com/我的云端硬盘/Final/daw-jepa-master/logs/jepa_r0.csv",
    "ema": "/Users/lucas/Library/CloudStorage/GoogleDrive-jwu14572@gmail.com/我的云端硬盘/Final/daw-jepa-master/logs/vits16_224-bs128-ep100-ema-20251206-214301/jepa_r0.csv",
    "instant": "/Users/lucas/Library/CloudStorage/GoogleDrive-jwu14572@gmail.com/我的云端硬盘/Final/daw-jepa-master/logs/vits16_224-bs128-ep100-instant-20251204-183358/jepa_r0.csv"
}


PROJECT = "ijepa_daw_compare"
GROUP = "epoch_loss"

for name, path in csv_files.items():

    print(f"Uploading {name} from {path}")

    # 1. Start W&B run
    run = wandb.init(
        project=PROJECT,
        group=GROUP,
        name=f"{name}_epoch_curve"
    )

    # 2. Load CSV
    df = pd.read_csv(path)

    # 3. Compute epoch-level average
    epoch_loss = df.groupby("epoch")["loss"].mean().reset_index()

    # 4. Upload: W&B will draw the line (one point per epoch)
    for _, row in epoch_loss.iterrows():
        run.log({
            "epoch": int(row["epoch"]),
            "epoch_loss": float(row["loss"])
        })

    run.finish()

print("\n🎉 Done! Go to W&B → Charts → epoch_loss → overlay = 自动三条曲线")
