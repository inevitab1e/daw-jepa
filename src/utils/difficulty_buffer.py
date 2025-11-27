import torch


class DifficultyBuffer:
    def __init__(
            self,
            num_samples,
            ema_alpha=0.9,
            gamma=1.0,
            w_min=0.5,
            w_max=2.0,
            warmup_epochs=0,
            mode="ema",  # "instant" / "ema"
            device="cpu",
    ):
        self.num_samples = num_samples
        self.ema_alpha = ema_alpha
        self.gamma = gamma
        self.w_min = w_min
        self.w_max = w_max
        self.warmup_epochs = warmup_epochs
        self.mode = mode

        self.buffer = torch.zeros(num_samples, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, indices, losses):
        """
        indices: [B] long
        losses:  [B] float (当前 step 的 per-sample loss)
        """
        if self.mode == "instant":
            # 只看当前 loss
            self.buffer[indices] = losses
        else:
            # EMA：历史 + 当前
            old = self.buffer[indices]
            new = self.ema_alpha * old + (1.0 - self.ema_alpha) * losses
            self.buffer[indices] = new

    @torch.no_grad()
    def get_weights(self, indices, epoch):
        # warmup 阶段不启用 DAW
        if epoch < self.warmup_epochs:
            return torch.ones(
                indices.shape[0], dtype=torch.float32, device=self.buffer.device
            )

        d = self.buffer[indices]  # [B]

        # 标准化一下，防止 exp 爆炸
        d_mean = d.mean()
        d_std = d.std()
        if d_std > 0:
            d_norm = (d - d_mean) / (d_std + 1e-6)
        else:
            d_norm = torch.zeros_like(d)

        w = torch.exp(self.gamma * d_norm)

        w = torch.clamp(w, self.w_min, self.w_max)

        w = w * (w.numel() / (w.sum() + 1e-6))

        return w
