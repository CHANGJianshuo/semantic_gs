#!/usr/bin/env python3
"""
Baseline Evaluation Script for AMtown02 Dataset

This script evaluates the baseline model on a held-out test set using:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

Note: This requires a Gaussian Splatting viewer/renderer to render images from the trained model.
For now, this is a template script. Actual evaluation will be completed when:
1. Test set is defined
2. Rendering tool is set up
"""

import numpy as np
import json
from pathlib import Path
from datetime import date
import argparse

def define_test_set(total_images=1380, test_ratio=0.1, seed=42):
    """
    Define test set from AMtown02 dataset.
    
    Args:
        total_images: Total number of images in dataset
        test_ratio: Ratio of images to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        List of test image indices
    """
    np.random.seed(seed)
    num_test = int(total_images * test_ratio)
    
    # Sample evenly across the sequence
    test_indices = np.linspace(0, total_images-1, num_test, dtype=int)
    
    return sorted(test_indices.tolist())

def calculate_metrics(rendered_dir: str, gt_dir: str) -> dict:
    """
    Calculate PSNR, SSIM, LPIPS for all test images.
    
    Args:
        rendered_dir: Directory containing rendered images from baseline model
        gt_dir: Directory containing ground truth test images
    
    Returns:
        Dictionary with mean metrics
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        import torch
        import lpips
        import cv2
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Install: pip install scikit-image torch lpips opencv-python")
        return None
    
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='vgg')
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
    
    rendered_files = sorted(Path(rendered_dir).glob('*.png'))
    if len(rendered_files) == 0:
        print(f"No rendered images found in {rendered_dir}")
        return None
    
    psnr_list, ssim_list, lpips_list = [], [], []
    
    print(f"\n{'='*60}")
    print(f"Evaluating {len(rendered_files)} test images...")
    print(f"{'='*60}\n")
    
    for i, rendered_path in enumerate(rendered_files):
        gt_path = Path(gt_dir) / rendered_path.name
        
        if not gt_path.exists():
            print(f"Warning: Ground truth not found for {rendered_path.name}")
            continue
        
        # Load images (BGR to RGB)
        rendered = cv2.imread(str(rendered_path))
        rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        
        gt = cv2.imread(str(gt_path))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        
        # Check dimensions match
        if rendered.shape != gt.shape:
            print(f"Warning: Shape mismatch for {rendered_path.name}")
            print(f"  Rendered: {rendered.shape}, GT: {gt.shape}")
            continue
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(gt, rendered, data_range=255)
        psnr_list.append(psnr)
        
        # Calculate SSIM
        ssim = structural_similarity(gt, rendered, channel_axis=2, data_range=255)
        ssim_list.append(ssim)
        
        # Calculate LPIPS
        rendered_t = torch.from_numpy(rendered).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1
        gt_t = torch.from_numpy(gt).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1
        
        if torch.cuda.is_available():
            rendered_t = rendered_t.cuda()
            gt_t = gt_t.cuda()
        
        with torch.no_grad():
            lpips_val = lpips_model(rendered_t, gt_t).item()
        lpips_list.append(lpips_val)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(rendered_files):
            print(f"Processed {i + 1}/{len(rendered_files)} images")
    
    if len(psnr_list) == 0:
        print("No valid image pairs found for evaluation")
        return None
    
    metrics = {
        'psnr': round(np.mean(psnr_list), 2),
        'ssim': round(np.mean(ssim_list), 4),
        'lpips': round(np.mean(lpips_list), 4),
        'num_images': len(psnr_list)
    }
    
    print(f"\n{'='*60}")
    print(f"Baseline Evaluation Results ({metrics['num_images']} images)")
    print(f"{'='*60}")
    print(f"PSNR:  {metrics['psnr']} dB")
    print(f"SSIM:  {metrics['ssim']}")
    print(f"LPIPS: {metrics['lpips']}")
    print(f"{'='*60}\n")
    
    return metrics

def save_baseline_results(metrics: dict, output_path: str):
    """Save baseline evaluation results to JSON"""
    baseline = {
        "model": "Baseline",
        "group_id": "Baseline_CPU_300iter",
        "group_name": "Official Baseline (CPU, 300 iterations)",
        "metrics": {
            "psnr": metrics['psnr'],
            "ssim": metrics['ssim'],
            "lpips": metrics['lpips']
        },
        "training_config": {
            "iterations": 300,
            "device": "CPU",
            "downscale_factor": 4,
            "training_time_minutes": 25,
            "num_training_images": 1380,
            "num_test_images": metrics['num_images']
        },
        "evaluation_date": str(date.today())
    }
    
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=4)
    
    print(f"✅ Baseline results saved to: {output_path}\n")
    print(json.dumps(baseline, indent=4))

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline model on test set')
    parser.add_argument('--rendered', type=str, help='Directory with rendered test images')
    parser.add_argument('--gt', type=str, help='Directory with ground truth test images')
    parser.add_argument('--output', type=str, default='baseline_results.json', 
                       help='Output JSON file')
    parser.add_argument('--test-set-only', action='store_true',
                       help='Only generate test set definition')
    
    args = parser.parse_args()
    
    # Define test set
    print("\n" + "="*60)
    print("AMtown02 Test Set Definition")
    print("="*60)
    
    test_indices = define_test_set(total_images=1380, test_ratio=0.1, seed=42)
    
    print(f"\nTest set: {len(test_indices)} images (10% of dataset)")
    print(f"Test indices (every ~10th image): {test_indices[:10]}... (first 10 shown)")
    
    # Save test set
    test_set_file = Path(__file__).parent / "test_set_indices.json"
    with open(test_set_file, 'w') as f:
        json.dump({"test_indices": test_indices, "total_images": 1380, "test_ratio": 0.1}, f, indent=4)
    print(f"✅ Test set saved to: {test_set_file}")
    
    if args.test_set_only:
        return
    
    # Evaluate if directories provided
    if args.rendered and args.gt:
        metrics = calculate_metrics(args.rendered, args.gt)
        if metrics:
            save_baseline_results(metrics, args.output)
    else:
        print("\n" + "="*60)
        print("⚠️  Evaluation not performed")
        print("="*60)
        print("To evaluate baseline, provide rendered and ground truth directories:")
        print(f"  python3 {Path(__file__).name} --rendered <rendered_dir> --gt <gt_dir>")
        print("\nNote: You need to render test images using a Gaussian Splatting viewer first.")
        print("="*60)

if __name__ == "__main__":
    main()

