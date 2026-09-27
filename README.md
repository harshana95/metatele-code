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

`weights/DAPE.pth` (DAPE condition model weights) is bundled in this repo.

Download the RAM vision-language model weights and place them at `weights/ram_swin_large_14m.pth`:

```bash
curl -L -o weights/ram_swin_large_14m.pth \
  https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
```

SD 2.1 (`sd2-community/stable-diffusion-2-1`) is downloaded automatically from HuggingFace Hub on first run.

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

Quick validation run:

```bash
python trainer.py -test -opt configs/train.yml
```

Outputs are saved under `./experiments/MetaZoom/`.

## Inference

1. Pick a config under `configs/infer_*.yml` (they differ by text-prompt extractor: `dape`, `florence`, `gemma`, `qwen`, `null`) and set `path.resume_from_path` to your trained experiment directory.
2. Run:

```bash
python trainer.py -infer -opt configs/infer_dape.yml
```

Or run the configs preselected in `scripts/infer.sh`:

```bash
bash scripts/infer.sh
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
@article{Weligampola2026MetaTele,
  title     = {MetaTele: compact refractive metasurface computational telephoto camera},
  author    = {Weligampola, Harshana and Chen, Yuanrui and Gnanasambandam, Abhiram
               and Godaliyadda, Dilshan and Sheikh, Hamid and Chan, Stanley
               and Guo, Qi},
  journal   = {Optics Express},
  volume    = {34},
  number    = {18},
  pages     = {34880--34897},
  year      = {2026},
  publisher = {Optica Publishing Group},
  url       = {https://opg.optica.org/oe/fulltext.cfm?uri=oe-34-18-34880}
}
```

## License

MIT License. See [LICENSE](LICENSE).
