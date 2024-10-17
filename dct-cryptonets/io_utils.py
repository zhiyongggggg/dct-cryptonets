import argparse
from models import backbone


model_dict = dict(
    ResNet20=backbone.ResNet20,
    ResNet20qat=backbone.ResNet20QAT,
    ResNet18=backbone.ResNet18,
    ResNet18qat=backbone.ResNet18QAT,
)


def parse_args(script):
    parser = argparse.ArgumentParser(
        description=f"DCT-CryptoNets ({'Training' if script == 'train' else 'Homomorphic Evaluation'})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_group = parser.add_argument_group('Default arguments')
    default_group.add_argument('--dataset', default='cifar10',
                               choices=['cifar10', 'ImageNet'],
                               help='Choose image dataset')
    default_group.add_argument('--model', default='ResNet18qat',
                               choices=['ResNet20', 'ResNet20qat',           # DCT-CryptoNets ResNet20
                                        'ResNet18', 'ResNet18qat'],          # DCT-CryptoNets ResNet18
                               help='Choose model architecture')
    default_group.add_argument('--num_classes', default=10, type=int, help='Number of prediction classes')
    default_group.add_argument('--dataset_path', metavar='PATH', help='Path to directory with dataset')
    default_group.add_argument('--save_path', metavar='PATH', help='Path to parent directory to save checkpoints')
    default_group.add_argument('--train_aug', action='store_true',
                               help='Perform data augmentation during training? (flag)')
    default_group.add_argument('--dct_status', action='store_true', help='Is this a DCT-based model? (flag)')
    default_group.add_argument('--channels', default=64, type=int,
                               choices=[3, 6, 24, 48, 64, 192],
                               help='If training a DCT-based model, choose the top-n amount of low-frequency DCT'
                                    ' components. Otherwise if training RGB-based model, use 3.')
    default_group.add_argument('--filter_size', default=8, type=int, help='DCT filter size')
    default_group.add_argument('--image_size', default=32, type=int, help='Size of non-DCT spatial dimensions')
    default_group.add_argument('--image_size_dct', default=56, type=int, help='Size of DCT spatial dimensions')
    default_group.add_argument('--dct_pattern', default='default', type=str,
                               choices=['default', 'square', 'triangle', 'learned'],
                               help='DCT subset pattern')
    default_group.add_argument('--bit_width', default=4, type=int, help='Quantization bit-width')
    default_group.add_argument('--dropout', default=None, type=float, help="Fraction of fc layer to dropout")
    default_group.add_argument('--verbose', default=True, type=bool, help='Verbose log outputs')

    if script == 'train':
        train_group = parser.add_argument_group('Training arguments')
        train_group.add_argument('--save_freq', default=5, type=int, help='Save frequency')
        train_group.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        train_group.add_argument('--stop_epoch', default=400, type=int, help='Stopping epoch')
        train_group.add_argument('--resume', default='', type=str, metavar='PATH',
                                 help='path to latest checkpoint (default: none)')
        train_group.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], help='Type of optimizer')
        train_group.add_argument('--lr', default=0.001, type=float, help='learning rate')
        train_group.add_argument('--weight_decay', default=1e-5, type=float, help='optimizer regularization')
        train_group.add_argument('--momentum', default=0.9, type=float, help='optimizer momentum for SGD')
        train_group.add_argument('--grad_clip_value', default=None, type=float, help='Value to clip gradients')
        train_group.add_argument('--grad_clip_norm', default=None, type=float, help='Max norm to clip gradients')
        train_group.add_argument('--batch_size', default=16, type=int, help='batch size ')
        train_group.add_argument('--test_batch_size', default=2, type=int, help='batch size ')
        train_group.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule')
        train_group.add_argument('--schedule', type=int, nargs='+', default=None,
                                 help='Manually decrease learning rate at these epochs')
        train_group.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                                 help='Directory to save model checkpoints')
        train_group.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')

    elif script == 'homomorphic_eval':
        homomorphic_group = parser.add_argument_group('Homomorphic evaluation arguments')
        homomorphic_group.add_argument('--checkpoint_path', type=str, help='Filepath to checkpoint')
        homomorphic_group.add_argument('--calib_batch_size', default=64, type=int,
                                       help='Batch size used for post-training quantization calibration')
        homomorphic_group.add_argument('--test_batch_size', default=1, type=int, help='Inference batch size')
        homomorphic_group.add_argument('--test_subset', default=1, type=int,
                                       help='Number of images to perform inference on')
        homomorphic_group.add_argument('--fhe_mode', default='simulate', type=str,
                                       choices=['simulate', 'execute'],
                                       help='Use FHE simulator (for accuracy) or execute full FHE model (for latency)')
        homomorphic_group.add_argument('--rounding_threshold_bits', default=6, type=int,
                                       help='Scaling factor to remove least significant bits')
        homomorphic_group.add_argument('--n_bits', default=5, type=int, help='Bit-width of homomorphic circuit')
        homomorphic_group.add_argument('--p_error', default=0.01, type=float, help='PBS error probability')
        homomorphic_group.add_argument('--reliability_test', default=True,
                                       help='Perform accuracy reliability analysis over a range of 20 random subsets?')

    else:
       raise ValueError('Unknown script')

    return parser.parse_args()
