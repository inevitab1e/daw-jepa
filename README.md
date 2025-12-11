# DAW-JEPA: Difficulty-Aware Weighting for I-JEPA Pretraining

This repository is a fork of Meta's official **I-JEPA** implementation and adds a **Difficulty-Aware Weighting (DAW)** mechanism on top of the original I-JEPA pretraining objective.

The project pre-trains a ViT-S/16 encoder on **ImageNet-100**, and evaluates the learned representation on **STL10** and **CIFAR-10** using:
- Linear probing on top of a frozen encoder
- k-NN classification on frozen features

All experiments in this repo were run on a **single-GPU environment** (Google Colab GPU runtime).

---

## 1. Project Overview

**Goal.** Improve I-JEPA’s representation quality by dynamically reweighting training samples based on their current “difficulty”, rather than treating all images equally.

**High-level idea.**

1. Pre-train I-JEPA on ImageNet-100 using a ViT-S/16 backbone.
2. Track the per-image reconstruction difficulty (per-sample loss) during pretraining.
3. Use a **difficulty buffer** to assign higher weights to harder images and lower weights to easier ones.
4. Evaluate the frozen encoder on STL10 and CIFAR-10 via linear probes and k-NN classification.

This repo keeps all the original I-JEPA components (masking, encoder/predictor architecture, training loop), and adds DAW in a minimal, controlled way so that baseline and DAW variants are directly comparable.

---

## 2. Method: Difficulty-Aware Weighting (DAW)

The main methodological contribution is implemented in:

- `src/utils/difficulty_buffer.py`
- `src/train.py` (integration into the training loop)
- `configs/in100_vits16_ep100_daw*.yaml` (configuration)

### 2.1 Difficulty Buffer

For each training image, we maintain a scalar **difficulty score** based on its per-sample prediction loss:

- The JEPA loss is computed as a **Smooth L1 loss** between target features `h` and predicted features `z`:

  $$\text{diff} = \text{SmoothL1}(z, h) \quad \text{(element-wise, no reduction)}$$

- We then average over all spatial and mask dimensions to obtain a **per-sample loss**:

  $$\ell_i = \text{mean}(\text{diff}_i)$$

- Because I-JEPA uses multiple target masks per image, we:
  - group per-sample losses back to per-image losses  
  - average them to obtain a final **per-image difficulty**

The **difficulty buffer** (`DifficultyBuffer`) stores a scalar difficulty for each sample index.

Two update modes are supported:

- `mode="instant"`:  
  `buffer[idx] = current_loss`
- `mode="ema"`:  
  `buffer[idx] = ema_alpha * old + (1 - ema_alpha) * current_loss`

This is controlled by the `daw` block in the YAML configs, e.g.:

```yaml
daw:
  enabled: true
  mode: ema         # "ema" or "instant"
  ema_alpha: 0.7
  gamma: 0.3
  w_min: 0.8
  w_max: 1.2
  warmup_epochs: 20
```

## 3. Reproducing Our Runs in Colab

For convenience, we provide a Colab-style notebook that contains the exact commands and configuration we used to run all experiments in this repository.  
See:

- `notebooks/DAW_JEPA_Colab_Clean.ipynb`

This notebook walks through:
- Setting up the environment in Google Colab
- Launching I-JEPA pretraining with DAW enabled
- Running linear probing and k-NN evaluation on STL10 and CIFAR-10

You can follow it cell-by-cell to reproduce our results on a single-GPU Colab runtime.
