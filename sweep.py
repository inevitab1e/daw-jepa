import os
import yaml
import copy
import subprocess

BASE_CONFIG = "configs/in100_vits16_ep100_daw.yaml"
SWEEP_DIR = "configs/sweep/"
os.makedirs(SWEEP_DIR, exist_ok=True)

MODES = ["instant"]
GAMMAS = [0.5]

# 固定参数
W_MIN, W_MAX = 0.8, 1.2
WARMUP = 10
EMA_ALPHA = 0.7


def load_base():
    with open(BASE_CONFIG, "r") as f:
        return yaml.safe_load(f)


def save_cfg(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def run_training(config_path):
    cmd = f"python main.py --fname {config_path} --devices cuda:0"
    print(f"\n=========================\nRunning: {cmd}\n=========================\n")
    subprocess.run(cmd, shell=True)


def main():
    base = load_base()

    for mode in MODES:
        for gamma in GAMMAS:
            cfg = copy.deepcopy(base)

            # 修改 DAW 配置
            cfg["daw"]["enabled"] = True
            cfg["daw"]["mode"] = mode
            cfg["daw"]["gamma"] = float(gamma)
            cfg["daw"]["w_min"] = W_MIN
            cfg["daw"]["w_max"] = W_MAX
            cfg["daw"]["warmup_epochs"] = WARMUP
            cfg["daw"]["ema_alpha"] = EMA_ALPHA

            # 生成 sweep 配置文件名
            name = f"in100_daw_{mode}_g{gamma}.yaml"
            config_path = os.path.join(SWEEP_DIR, name)

            save_cfg(cfg, config_path)
            print(f"Generated {config_path}")

            run_training(config_path)


if __name__ == "__main__":
    main()
