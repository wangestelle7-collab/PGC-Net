import argparse


def get_train_options():
    """Get training parameter configuration"""
    parser = argparse.ArgumentParser(description='GAN-Based Medical Image Translation Training Options')

    # Data path parameters
    parser.add_argument('--source_image_root', type=str, default='./dataset/pet/',
                        help='Root path of source image data')
    parser.add_argument('--target_image_root', type=str, default='./dataset/ct/',
                        help='Root path of target image data')
    parser.add_argument('--mask_root', type=str, default='./dataset/mask/',
                        help='Root path of mask data')
    parser.add_argument('--dataset_name', type=str, default='dataset',
                        help='Dataset name')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='Learning rate decay rate')
    parser.add_argument('--decay_epoch', type=int, default=20,
                        help='Starting epoch for learning rate decay')
    parser.add_argument('--trainsize', type=int, default=256,
                        help='Training image size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    # Model parameters
    parser.add_argument('--save_model_path', type=str, default='./saved_models/',
                        help='Model save path')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Model save frequency (save every N epochs)')
    parser.add_argument('--log_path', type=str, default='./logs/',
                        help='Log save path')

    # Loss function weights (adjust as needed)
    parser.add_argument('--gan_weight', type=float, default=1.0,
                        help='GAN loss weight')
    parser.add_argument('--pix_weight', type=float, default=100.0,
                        help='Pixel loss weight')
    parser.add_argument('--feature_weight', type=float, default=1.0,
                        help='Feature loss weight')
    parser.add_argument('--mid_weight', type=float, default=50.0,
                        help='Intermediate output loss weight')
    parser.add_argument('--str_weight', type=float, default=1.0,
                        help='Structure loss weight')
    parser.add_argument('--tex_weight', type=float, default=50.0,
                        help='Texture loss weight')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU device ID (separate multiple IDs with commas)')

    return parser.parse_args()