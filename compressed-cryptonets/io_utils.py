import numpy as np
import os
import glob
import argparse
from models import backbone
from models import mobilenetv2


model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4_QAT=backbone.Conv4_QAT,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet20=backbone.ResNet20,
    ResNet20qat=backbone.ResNet20QAT,
    ResNet10=backbone.ResNet10,
    ResNet10qat=backbone.ResNet10QAT,
    ResNet18=backbone.ResNet18,
    ResNet18qat=backbone.ResNet18QAT,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101,
    MobileNetv2=mobilenetv2.mobilenetv2,
    MobileNetv2DCT=mobilenetv2.mobilenetv2dct_subset_woinp,
)


def parse_args(script):
    parser = argparse.ArgumentParser(description=f'Frequency CryptoNets Script {script}')
    parser.add_argument('--dataset', default='cifar10', help='miniImagenet/cifar10/mnist/chest_xray')
    parser.add_argument('--num_classes', default=200, type=int,
                        help='total number of classes in softmax, only used in baseline')  # make it larger than the maximum label value in base class
    parser.add_argument('--dataset_path', required=False, help='Path to directory with dataset')
    parser.add_argument('--model', default='WideResNet28_10', help='model: WideResNet28_10 /Conv{4|6} /ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method', default='baseline++', help='baseline++/rotation/manifold_mixup/S2M2_R') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way', default=5, type=int, help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way', default=5, type=int, help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug', action='store_true', help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--channels', default=24, type=int, help='DCT channels 6/24/64')
    parser.add_argument('--dct_status', action='store_true', help='true/false')
    parser.add_argument('--filter_size', default=8, type=int, help='DCT filter size (default=8x8)')
    parser.add_argument('--image_size', default=84, type=int, help='Size of non-DCT spatial dimensions 84/224/448')
    parser.add_argument('--image_size_dct', default=56, type=int, help='Size of DCT spatial dimensions 56/28')
    parser.add_argument('--dct_pattern', default='square', type=str, choices=['default', 'square', 'triangle', 'learned'], help='DCT subset pattern')
    parser.add_argument('--bit_width', default=4, type=int, help='Quantization bit-width (2^n)')
    parser.add_argument('--pruning', default=0.1, type=float, help="Fraction of randomly pruned weights")
    parser.add_argument('--dropout', default=None, type=float, help="Fraction of fc (classifier) layer to dropout")
    if script == 'train':
        parser.add_argument('--save_freq', default=5, type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=400, type=int, help='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], help='Type of optimizer')
        parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
        parser.add_argument('--weight_decay', default=1e-5, type=float, help='optimizer regularization')
        parser.add_argument('--momentum', default=0.9, type=float, help='optimizer momentum for SGD')
        parser.add_argument('--grad_clip_value', default=None, type=float, help='Value to clip gradients')
        parser.add_argument('--grad_clip_norm', default=None, type=float, help='Max norm to clip gradients')
        parser.add_argument('--batch_size', default=16, type=int, help='batch size ')
        parser.add_argument('--test_batch_size', default=2, type=int, help='batch size ')
        parser.add_argument('--alpha', default=2.0, type=float, help='for manifold_mixup or S2M2 training ')
        parser.add_argument('--warmup', action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--schedule', type=int, nargs='+', default=None,
                            help='Manually decrease learning rate at these epochs. '
                                 'If none specified then using automatic ReduceLROnPlateau method')
        parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                            help='Directory to save model checkpoints')
        parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    elif script == 'save_features':
        parser.add_argument('--split', default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int, help='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--checkpoint_path', type=str, help='Filepath to checkpoint')
        parser.add_argument('--test_batch_size', default=2, type=int, help='batch size ')
        parser.add_argument('--split', default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int, help='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--num_classes', default=200, type=int, help='total number of classes')
    elif script == 'homomorphic_eval':
        parser.add_argument('--checkpoint_path', type=str, help='Filepath to checkpoint')
        parser.add_argument('--split', default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--batch_size', default=64, type=int, help='batch size used for post-training quantization calibration')
        parser.add_argument('--test_batch_size', default=1, type=int, help='inference batch size')
        parser.add_argument('--test_subset', default=1, type=int, help='number of images to perform inference on')
        parser.add_argument('--fhe_mode', default='simulate', type=str, help='Use FHE simulator or execute (simulate/execute)')
        parser.add_argument('--rounding_threshold_bits', default=6, type=int)
        parser.add_argument('--n_bits', default=5, type=int)
        parser.add_argument('--p_error', default=0.01, type=float)
    else:
       raise ValueError('Unknown script')
    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
