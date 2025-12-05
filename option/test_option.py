import argparse


def get_test_options():
    """Get test configuration parameters"""
    parser = argparse.ArgumentParser(description='GAN-Based Medical Image Translation Test Options')

    # 数据路径参数
    parser.add_argument('--source_image_root', type=str, default='./dataset/source/pettest/',
                        help='Test source image root path')
    parser.add_argument('--target_image_root', type=str, default='./dataset/target/cttest/',
                        help='Test target image root path (for reference)')
    parser.add_argument('--dataset_name', type=str, default='dataset',
                        help='Dataset name (same as training)')

    # 模型加载参数
    parser.add_argument('--generator_path', type=str, required=True,
                        help='Pretrained Generator model path (e.g., ./saved_models/Best_Generator_50.pth)')
    parser.add_argument('--generator_ct_feature_path', type=str, required=True,
                        help='Pretrained Generator_Ct_Feature model path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for testing (cuda/cpu)')

    # 测试配置参数
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing (usually 1 for stable inference)')
    parser.add_argument('--testsize', type=int, default=256,
                        help='Test image size (must match training size)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    # 输出配置参数
    parser.add_argument('--output_path', type=str, default='./test_results/',
                        help='Path to save test output (generated images + metrics)')
    parser.add_argument('--save_images', type=bool, default=True,
                        help='Whether to save generated images')
    parser.add_argument('--log_name', type=str, default='test_log.txt',
                        help='Test log filename')
    parser.add_argument('--metric_save_path', type=str, default='test_metrics.csv',
                        help='Path to save metrics in CSV format')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU device ID (only valid if device=cuda)')

    opt = parser.parse_args()
    return opt