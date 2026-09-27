import os

# Headless-safe backends (no X11/Qt display required on compute nodes)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import pyiqa
import pandas as pd
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

ALL_METRICS = [
    'psnr', 'ssim', 
    'lpips', 'dists', 'fid',
    'niqe', 'musiq', 'maniqa', 'clipiqa',
    'ciede2000', 
]
SUMMARY_METRIC_ORDER = [
    'psnr', 'ssim', 'lpips', 'dists', 'fid', 'niqe', 'musiq', 'maniqa', 'clipiqa', 'ciede2000',
]


def parse_metrics(metrics_str):
    if metrics_str is None:
        return list(ALL_METRICS)
    selected = [m.strip() for m in metrics_str.split(',') if m.strip()]
    unknown = set(selected) - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}. Choose from: {ALL_METRICS}")
    return selected


def load_images_from_folder(folder, image_size):
    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))
    ])
    images = []
    for fname in image_files:
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        img = img.resize((image_size, image_size))
        images.append(img)
    return images, image_files


def match_and_resize_images(gt_images, pred_images):
    matched_gt_images = []
    matched_pred_images = []
    for gt_img, pred_img in zip(gt_images, pred_images):
        gt_w, gt_h = gt_img.size
        pred_w, pred_h = pred_img.size
        if (gt_w, gt_h) != (pred_w, pred_h):
            new_w = max(gt_w, pred_w)
            new_h = max(gt_h, pred_h)
            gt_img = gt_img.resize((new_w, new_h))
            pred_img = pred_img.resize((new_w, new_h))
        matched_gt_images.append(gt_img)
        matched_pred_images.append(pred_img)
    return matched_gt_images, matched_pred_images


def save_image_pairs(gt_images, pred_images, pred_folder):
    pairs_folder = os.path.join(pred_folder, "pairs")
    os.makedirs(pairs_folder, exist_ok=True)
    print(f"Saving pairs in {pairs_folder}")

    for i, (gt_img, pred_img) in enumerate(zip(gt_images, pred_images)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gt_img.transpose([1, 2, 0]))
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        axes[1].imshow(pred_img.transpose([1, 2, 0]))
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        plt.tight_layout()
        save_path = os.path.join(pairs_folder, f"pair_{i + 1}.png")
        plt.savefig(save_path)
        plt.close(fig)


def compute_mean_ciede2000(gt_tensor, pred_tensor):
    """Mean CIEDE2000 (ΔE) over pixels between two RGB images in [0, 1]."""
    gt_np = gt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if gt_np.shape[-1] == 1:
        gt_np = np.repeat(gt_np, 3, axis=-1)
        pred_np = np.repeat(pred_np, 3, axis=-1)
    lab_gt = rgb2lab(gt_np)
    lab_pred = rgb2lab(pred_np)
    return float(np.mean(deltaE_ciede2000(lab_gt, lab_pred)))


def compute_batch_psnr(pred_tensors, gt_tensors):
    mse = torch.mean((pred_tensors - gt_tensors) ** 2, dim=(1, 2, 3))
    return (10 * torch.log10(1.0 / mse.clamp(min=1e-10))).detach().cpu().tolist()


def compute_batch_lpips(lpips_metric, pred_tensors, gt_tensors):
    scores = lpips_metric(pred_tensors * 2 - 1, gt_tensors * 2 - 1)
    flat = scores.detach().cpu().view(-1)
    if len(flat) == len(pred_tensors):
        return flat.tolist()

    per_image = []
    for i in range(len(pred_tensors)):
        score = lpips_metric(pred_tensors[i:i + 1] * 2 - 1, gt_tensors[i:i + 1] * 2 - 1)
        per_image.append(score.detach().cpu().view(-1).item())
    return per_image


def normalize_metric_lengths(values, summary_keys):
    n = len(values['name'])
    for key in summary_keys:
        if key not in values:
            values[key] = [float('nan')] * n
            continue
        series = values[key]
        if len(series) == n:
            continue
        if len(series) == 1:
            values[key] = series * n
            continue
        raise ValueError(
            f"Metric '{key}' has length {len(series)}, expected {n} "
            f"(name column length)."
        )


def compute_batch_ssim(ssim_metric, pred_tensors, gt_tensors):
    scores = []
    for i in range(len(pred_tensors)):
        scores.append(ssim_metric(pred_tensors[i:i + 1], gt_tensors[i:i + 1]).item())
    return scores


def run_pyiqa_metric(metric_fn, metric_name, pred_tensors, gt_tensors):
    try:
        if metric_name == 'dists':
            batch_scores = metric_fn(pred_tensors, gt_tensors)
        else:
            batch_scores = metric_fn(pred_tensors)
        if isinstance(batch_scores, torch.Tensor):
            if batch_scores.ndim == 0:
                return [batch_scores.item()] * len(pred_tensors)
            flat = batch_scores.detach().cpu().view(-1)
            if len(flat) == len(pred_tensors):
                return flat.tolist()
    except Exception:
        pass

    scores = []
    for i in range(len(pred_tensors)):
        pred = pred_tensors[i:i + 1]
        if metric_name == 'dists':
            scores.append(metric_fn(pred, gt_tensors[i:i + 1]).item())
        else:
            scores.append(metric_fn(pred).item())
    return scores


def compute_batch_ciede2000(gt_tensors, pred_tensors):
    return [compute_mean_ciede2000(gt_tensors[i:i + 1], pred_tensors[i:i + 1]) for i in range(len(gt_tensors))]


class MetricSuite:
    """Load each metric model once and reuse across multiple prediction folders."""

    def __init__(self, device, metrics):
        self.device = device
        self.metrics = set(metrics)

        if 'ssim' in self.metrics:
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        if 'lpips' in self.metrics:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        if 'fid' in self.metrics:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

        self.pyiqa_metrics = {}
        for name in ('dists', 'niqe', 'musiq', 'maniqa', 'clipiqa'):
            if name in self.metrics:
                self.pyiqa_metrics[name] = pyiqa.create_metric(name, device=device)


@torch.no_grad()
def compute_metrics(
    gt_folder,
    pred_folder,
    image_size,
    device,
    name,
    suite,
    enabled_metrics,
    channel='none',
    channel_gt='none',
    save_pairs=False,
    save_outputs=False,
    results_file=None,
):
    gt_images, gt_names = load_images_from_folder(gt_folder, image_size)
    pred_images, _ = load_images_from_folder(pred_folder, image_size)
    n_images = min(len(gt_images), len(pred_images))
    if len(gt_images) != len(pred_images):
        print(
            f"Warning: {pred_folder} has {len(pred_images)} images but "
            f"GT has {len(gt_images)}; using first {n_images} pairs."
        )
    gt_images = gt_images[:n_images]
    pred_images = pred_images[:n_images]
    gt_names = gt_names[:n_images]

    if channel in ['r', 'g', 'b']:
        idx = 'rgb'.index(channel)
        gt_idx = 'rgb'.index(channel_gt)
        gt_images = [img.split()[gt_idx] for img in gt_images]
        pred_images = [img.split()[idx] for img in pred_images]

    gt_images, pred_images = match_and_resize_images(gt_images, pred_images)

    transform = transforms.Compose([transforms.ToTensor()])
    gt_tensors = torch.stack([transform(img) for img in gt_images]).to(device)
    pred_tensors = torch.stack([transform(img) for img in pred_images]).to(device)

    if save_pairs:
        save_image_pairs(gt_tensors.cpu().numpy(), pred_tensors.cpu().numpy(), pred_folder)
    if save_outputs:
        os.makedirs(os.path.join(pred_folder, 'out'), exist_ok=True)

    values = {'name': [os.path.basename(gt_name) for gt_name in gt_names]}

    if 'psnr' in suite.metrics:
        values['psnr'] = compute_batch_psnr(pred_tensors, gt_tensors)
    if 'ssim' in suite.metrics:
        values['ssim'] = compute_batch_ssim(suite.ssim, pred_tensors, gt_tensors)
    if 'lpips' in suite.metrics:
        values['lpips'] = compute_batch_lpips(suite.lpips, pred_tensors, gt_tensors)
    if 'ciede2000' in suite.metrics:
        values['ciede2000'] = compute_batch_ciede2000(gt_tensors, pred_tensors)

    for metric_name, metric_fn in suite.pyiqa_metrics.items():
        values[metric_name] = run_pyiqa_metric(metric_fn, metric_name, pred_tensors, gt_tensors)

    if 'fid' in suite.metrics:
        suite.fid.reset()
        suite.fid.update(gt_tensors, real=True)
        suite.fid.update(pred_tensors, real=False)
        values['fid'] = [suite.fid.compute().item()] * len(pred_tensors)

    if save_outputs:
        for idx, pred_img in enumerate(pred_images):
            pred_img.save(os.path.join(pred_folder, 'out', f"{name}_{idx + 1}.png"))

    for key in values:
        for i in range(len(values[key])):
            if isinstance(values[key][i], torch.Tensor):
                values[key][i] = values[key][i].item()

    summary_keys = [key for key in SUMMARY_METRIC_ORDER if key in enabled_metrics]
    normalize_metric_lengths(values, summary_keys)
    to_print = f"{pred_folder}"
    for key in summary_keys:
        value = float(np.mean(values[key]))
        to_print += f" & {value:.4f}"
    to_print += " \\\\"
    print(to_print)

    if results_file is None:
        results_file = os.path.join(gt_folder, "results.txt")
    with open(results_file, "a") as f:
        f.write(f"{to_print}\n")

    df = pd.DataFrame(values, columns=['name'] + summary_keys)
    df.to_csv(os.path.join(pred_folder, "metrics.csv"), index=False)
    return values


@torch.no_grad()
def run_batch(
    gt_folder,
    pred_folders,
    image_size,
    device,
    enabled_metrics,
    results_file=None,
    channel='none',
    channel_gt='none',
    save_pairs=False,
    save_outputs=False,
):
    suite = MetricSuite(device, enabled_metrics)
    for pred_folder, name in pred_folders:
        print(f"========== {name} ==========")
        compute_metrics(
            gt_folder=gt_folder,
            pred_folder=pred_folder,
            image_size=image_size,
            device=device,
            name=name,
            suite=suite,
            enabled_metrics=enabled_metrics,
            channel=channel,
            channel_gt=channel_gt,
            save_pairs=save_pairs,
            save_outputs=save_outputs,
            results_file=results_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate image quality metrics between GT and predicted images.")
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to ground truth image folder')
    parser.add_argument('--pred_folder', type=str, default=None, help='Path to predicted image folder')
    parser.add_argument('--name', type=str, default='none', help='Method name for outputs')
    parser.add_argument('--methods', nargs='*', default=None, help='Baseline names under --results_dir')
    parser.add_argument('--results_dir', type=str, default=None, help='Root results directory for --methods')
    parser.add_argument('--image_size', type=int, required=True, help='Resize images to this square size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--metrics', type=str, default=None, help='Comma-separated metrics to compute')
    parser.add_argument('--channel', type=str, default='none', choices=['none', 'r', 'g', 'b'])
    parser.add_argument('--channel_gt', type=str, default='none', choices=['none', 'r', 'g', 'b'])
    parser.add_argument('--save_pairs', action='store_true', help='Save side-by-side GT/pred PNGs')
    parser.add_argument('--save_outputs', action='store_true', help='Save annotated prediction PNGs')
    parser.add_argument('--results_file', type=str, default=None, help='Path to append summary results')
    args = parser.parse_args()

    enabled_metrics = parse_metrics(args.metrics)

    if args.methods:
        if args.results_dir is None:
            parser.error('--results_dir is required when using --methods')
        pred_folders = [
            (os.path.join(args.results_dir, method), method)
            for method in args.methods
            if os.path.isdir(os.path.join(args.results_dir, method))
        ]
        missing = [
            method for method in args.methods
            if not os.path.isdir(os.path.join(args.results_dir, method))
        ]
        for method in missing:
            print(f"Skipping {method}: directory not found under {args.results_dir}")
        if not pred_folders:
            raise SystemExit("No valid method directories found.")
        run_batch(
            gt_folder=args.gt_folder,
            pred_folders=pred_folders,
            image_size=args.image_size,
            device=args.device,
            enabled_metrics=enabled_metrics,
            results_file=args.results_file,
            channel=args.channel,
            channel_gt=args.channel_gt,
            save_pairs=args.save_pairs,
            save_outputs=args.save_outputs,
        )
    else:
        if args.pred_folder is None:
            parser.error('--pred_folder is required unless --methods is provided')
        suite = MetricSuite(args.device, enabled_metrics)
        compute_metrics(
            gt_folder=args.gt_folder,
            pred_folder=args.pred_folder,
            image_size=args.image_size,
            device=args.device,
            name=args.name,
            suite=suite,
            enabled_metrics=enabled_metrics,
            channel=args.channel,
            channel_gt=args.channel_gt,
            save_pairs=args.save_pairs,
            save_outputs=args.save_outputs,
            results_file=args.results_file,
        )
