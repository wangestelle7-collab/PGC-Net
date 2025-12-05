from datetime import datetime
from models.PGC import Generator # Only need generator-related models
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import logging
import csv
from torch.utils.data import DataLoader
from datasets import ImageDatasetV2  # Same dataset class as training
from option.test_option import get_test_options
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# --------------------------
# Utility Functions
# --------------------------
def normalize_output(data):
    """Data normalization (consistent with training)"""
    data = data.clone()
    data = torch.clamp(data, -1.0, 1.0)
    return (data + 1.0) / 2.0


def tensor_to_image(tensor):
    """Convert torch tensor to PIL Image"""
    tensor = tensor.squeeze(0).cpu().detach().numpy()
    tensor = (tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(tensor)


def calculate_metrics(generated_img, target_img):
    """Calculate image quality metrics (PSNR, SSIM, MAE, MSE)"""
    # Convert tensors to numpy arrays (0-1 range)
    gen_np = generated_img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    tar_np = target_img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)

    # Ensure values are in [0, 1]
    gen_np = np.clip(gen_np, 0.0, 1.0)
    tar_np = np.clip(tar_np, 0.0, 1.0)

    # Calculate metrics
    psnr_val = psnr(tar_np, gen_np, data_range=1.0)
    ssim_val = ssim(tar_np, gen_np, data_range=1.0, channel_axis=2)
    mae_val = np.mean(np.abs(tar_np - gen_np))
    mse_val = np.mean((tar_np - gen_np) ** 2)

    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'MAE': mae_val,
        'MSE': mse_val
    }


def setup_logging(opt):
    """Setup logging configuration"""
    log_dir = os.path.join(opt.output_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, opt.log_name)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                                  datefmt='%Y-%m-%d %I:%M:%S %p')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('Test')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicate logs

    return logger


def load_pretrained_model(model, model_path, device, logger):
    """Load pretrained model weights with device compatibility"""
    try:
        # Load state dict with map_location to handle CPU/GPU mismatch
        state_dict = torch.load(model_path, map_location=device)

        # Remove 'module.' prefix if model was trained with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        logger.info(f"Successfully loaded model from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}", exc_info=True)
        raise SystemExit(1)


# --------------------------
# Main Test Function
# --------------------------
def test(opt):
    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(opt)

    # Log configuration
    logger.info("Starting model testing...")
    logger.info("Test Configuration Parameters:")
    for arg in vars(opt):
        logger.info(f'{arg}: {getattr(opt, arg)}')
    logger.info(f'Using device: {device}')

    # Create output directories
    img_output_dir = os.path.join(opt.output_path, 'generated_images')
    os.makedirs(img_output_dir, exist_ok=True)

    # --------------------------
    # Model Initialization
    # --------------------------
    # Initialize models (same as training)
    generator = Generator().to(device)

    # Load pretrained weights
    generator = load_pretrained_model(generator, opt.generator_path, device, logger)

    # Set models to evaluation mode
    generator.eval()
    logger.info("Models set to evaluation mode")

    # --------------------------
    # Data Loading (consistent with training)
    # --------------------------
    transforms_ = [
        transforms.Resize((opt.testsize, opt.testsize), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDatasetV2(opt.dataset_name, transforms_=transforms_, mode='test'),
        batch_size=opt.batch_size,
        shuffle=False,  # No shuffle for testing
        num_workers=opt.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    logger.info(f"Test dataset loaded: {len(dataloader.dataset)} samples, {len(dataloader)} batches")

    # --------------------------
    # Test Metrics Initialization
    # --------------------------
    total_metrics = {
        'PSNR': 0.0,
        'SSIM': 0.0,
        'MAE': 0.0,
        'MSE': 0.0
    }
    metric_list = []  # For saving individual sample metrics

    # --------------------------
    # Testing Loop
    # --------------------------
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_idx, batch in enumerate(dataloader):
            # Load data
            source_imgs = batch["A"].to(device)
            target_imgs = batch["B"].to(device)
            filenames = batch.get("filename", [f"sample_{batch_idx}_{i}" for i in range(opt.batch_size)])

            # Forward pass (same as training's generator output)
            guided_CT, pet_commonFeature, gen_CT, seg = generator(source_imgs)

            # Normalize outputs to [0, 1] range
            gen_CT_norm = normalize_output(gen_CT)
            target_imgs_norm = normalize_output(target_imgs)
            source_imgs_norm = normalize_output(source_imgs)

            # --------------------------
            # Calculate Metrics
            # --------------------------
            for i in range(source_imgs.size(0)):
                metrics = calculate_metrics(
                    gen_CT_norm[i:i + 1],  # Keep batch dimension
                    target_imgs_norm[i:i + 1]
                )

                # Accumulate total metrics
                for key in total_metrics:
                    total_metrics[key] += metrics[key]

                # Record individual sample metrics
                metric_list.append({
                    'filename': filenames[i],
                    'PSNR': metrics['PSNR'],
                    'SSIM': metrics['SSIM'],
                    'MAE': metrics['MAE'],
                    'MSE': metrics['MSE']
                })

                # --------------------------
                # Save Generated Images
                # --------------------------
                if opt.save_images:
                    # Create combined image (source + generated + target) for comparison
                    source_img = tensor_to_image(source_imgs_norm[i:i + 1])
                    gen_img = tensor_to_image(gen_CT_norm[i:i + 1])
                    target_img = tensor_to_image(target_imgs_norm[i:i + 1])

                    # Combine images horizontally
                    combined_width = source_img.width + gen_img.width + target_img.width
                    combined_height = max(source_img.height, gen_img.height, target_img.height)
                    combined_img = Image.new('RGB', (combined_width, combined_height))

                    combined_img.paste(source_img, (0, 0))
                    combined_img.paste(gen_img, (source_img.width, 0))
                    combined_img.paste(target_img, (source_img.width + gen_img.width, 0))

                    # Save combined image
                    save_filename = f"{os.path.splitext(filenames[i])[0]}_comparison.png"
                    save_path = os.path.join(img_output_dir, save_filename)
                    combined_img.save(save_path)

                    # Save individual generated image
                    gen_save_path = os.path.join(img_output_dir, f"{os.path.splitext(filenames[i])[0]}_generated.png")
                    gen_img.save(gen_save_path)

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                avg_metrics = {k: v / (batch_idx + 1) * opt.batch_size for k, v in total_metrics.items()}
                logger.info(
                    f"Batch [{batch_idx + 1}/{len(dataloader)}] | "
                    f"Avg PSNR: {avg_metrics['PSNR']:.2f} | "
                    f"Avg SSIM: {avg_metrics['SSIM']:.4f} | "
                    f"Avg MAE: {avg_metrics['MAE']:.6f} | "
                    f"Avg MSE: {avg_metrics['MSE']:.6f}"
                )

    # --------------------------
    # Calculate Average Metrics
    # --------------------------
    num_samples = len(dataloader.dataset)
    avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}

    # Log final results
    logger.info("\n=============================================")
    logger.info("Test Completion Report")
    logger.info("=============================================")
    logger.info(f"Total test samples: {num_samples}")
    logger.info(f"Average PSNR: {avg_metrics['PSNR']:.2f} dB")
    logger.info(f"Average SSIM: {avg_metrics['SSIM']:.4f}")
    logger.info(f"Average MAE: {avg_metrics['MAE']:.6f}")
    logger.info(f"Average MSE: {avg_metrics['MSE']:.6f}")
    logger.info("=============================================\n")

    # --------------------------
    # Save Metrics to CSV
    # --------------------------
    csv_path = os.path.join(opt.output_path, opt.metric_save_path)
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'PSNR', 'SSIM', 'MAE', 'MSE'])
            writer.writeheader()
            writer.writerows(metric_list)

            # Add average row
            writer.writerow({
                'filename': 'AVERAGE',
                'PSNR': avg_metrics['PSNR'],
                'SSIM': avg_metrics['SSIM'],
                'MAE': avg_metrics['MAE'],
                'MSE': avg_metrics['MSE']
            })
        logger.info(f"Metrics saved to CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to CSV: {str(e)}", exc_info=True)

    logger.info("Testing process completed successfully!")


if __name__ == '__main__':
    # Get test options
    opt = get_test_options()

    # Run test
    test(opt)
