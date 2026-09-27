# MetaZoom

Standalone training and inference code for MetaZoom image restoration using a dual-input one-step diffusion model on Stable Diffusion 2.1.

## Method

This repository trains `twodataset_model`, which restores images from paired inputs:

- **Color blur** (1× magnification) from a HuggingFace dataset
- **Mono blur** (5× magnification, green channel) from a second HuggingFace dataset

Key components:

- **LoRA fine-tuning** on SD 2.1 UNet and VAE (6-channel VAE input via concatenated latents)
- **T2IAdapter** with high-pass filtered mono input (`hpf_adapter_input`)
- **RAM + DAPE** for automatic image captioning / text conditioning
- **Losses:** L2, LPIPS, chroma (YCbCr), and VSD/KL (`lambda_*` in config)

> **Note:** The `train.loss: 1*MSE` field in the YAML config is **not used**. Actual training losses are controlled by `lambda_l2`, `lambda_lpips`, `lambda_chroma`, and `lambda_kl`.

## Setup

### 1. Environment

```bash
conda create -n metazoom python=3.11 -y
conda activate metazoom
pip install -r requirements.txt
```

PyTorch with CUDA should be installed for your GPU. Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Download pretrained weights

Place the following files in `weights/`:

| File | Description |
|------|-------------|
| `ram_swin_large_14m.pth` | RAM vision-language model weights |
| `DAPE.pth` | DAPE condition model weights |

SD 2.1 (`stabilityai/stable-diffusion-2-1`) is downloaded automatically from HuggingFace Hub on first run.

### 3. Datasets

Training data is loaded from HuggingFace Hub (downloaded automatically):

- `harshana95/quadratic_color_psfs_5db_updated_real_hybrid_Flickr2k_gt_v2_PCA_interp_file`
- `harshana95/quadratic_mono_psfs_5db_updated_real_hybrid_Flickr2k_gt_v2_PCA_interp_file`

## Training

Single GPU:

```bash
bash scripts/train.sh
```

Multi-GPU (Accelerate):

```bash
accelerate launch --num_processes 8 trainer.py -opt configs/train.yml
```

Slurm (Gilbreth):

```bash
sbatch scripts/slurm_train.sh
```

Quick validation run:

```bash
python trainer.py -test -opt configs/train.yml
```

Outputs are saved under `./experiments/MetaZoom/`.

## Inference

1. Set `path.resume_from_path` in `configs/infer.yml` to your trained experiment directory.
2. Run:

```bash
bash scripts/infer.sh
```

Or explicitly:

```bash
python trainer.py -infer -opt configs/infer.yml
```

## Metrics

Compute PSNR, SSIM, and LPIPS on saved outputs:

```bash
bash scripts/calculate_metrics.sh <pred_dir> <gt_dir>
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `timestep` | 999 | One-step diffusion timestep |
| `lambda_l2` | 10.0 | Pixel L2 loss weight |
| `lambda_lpips` | 2.0 | Perceptual loss weight |
| `lambda_chroma` | 2.0 | YCbCr chroma loss weight |
| `lambda_kl` | 1.0 | VSD/KL loss weight |
| `cfg_vsd` | 7.5 | Classifier-free guidance for VSD |
| `learning_rate` | 1e-5 | Adam learning rate |
| `batch_size` | 4 | Training batch size per GPU |
| `max_train_steps` | 700000 | Total training steps |

Mono dataset uses `select_channels: [False, True, False]` to keep only the green channel before normalization.

## Citation

```bibtex
@article{2026metatele,
  title={MetaTele: Compact Refractive Metasurface Computational Telephoto Camera},
  author={Weligampola, Harshana and Chen, Yuanrui and Gnanasambandam, Abhiram and Godaliyadda, Dilshan and Sheikh, Hamid R and Chan, Stanley H and Guo, Qi},
  journal={arXiv preprint arXiv:2604.07614},
  year={2026}
}
```

## License

See [LICENSE](LICENSE). Replace with your chosen license before public release.
