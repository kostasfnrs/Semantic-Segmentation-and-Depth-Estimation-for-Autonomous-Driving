import argparse
import os
import json

from source.datasets.definitions import SPLIT_VALID, SPLIT_TEST


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--name', type=str, required=True, help='Name for your run to easier identify it.')
    parser.add_argument(
        '--log_dir', type=expandpath, required=True, help='Place for artifacts and logs')
    parser.add_argument(
        '--dataset_root', type=expandpath, required=True, help='Path to dataset')
    parser.add_argument(
        '--resume', type=str, default=None, help='Resume training from checkpoint: path to a valid file')

    parser.add_argument(
        '--prepare_submission', type=str2bool, default=False,
        help='Run best model on RGB test data, pack the archive with predictions for the grader')

    parser.add_argument(
        '--num_epochs', type=int, default=16, help='Number of training epochs')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Number of samples in a batch for training')
    parser.add_argument(
        '--batch_size_validation', type=int, default=8, help='Number of samples in a batch for validation')

    parser.add_argument(
        '--num_bins', type=int, default=256, help='Task3: Number of bins for adaptive depth, be sure its sqrt is an int!')
    parser.add_argument(
        '--num_heads', type=int, default=8, help='Task3: Number of heads for self attention')
    parser.add_argument(
        '--expansion', type=int, default=2, help='Task3: Expansion in inverse bottleneck in MLP')
    parser.add_argument(
        '--num_transformer_layers', type=int, default=2, help='Task3: Number of layer for transformer layer')
    
    parser.add_argument(
        '--aug_input_crop_size', type=int, default=256, help='Training crop size')
    parser.add_argument(
        '--aug_geom_scale_min', type=float, default=1.0, help='Augmentation: lower bound of scale')
    parser.add_argument(
        '--aug_geom_scale_max', type=float, default=1.0, help='Augmentation: upper bound of scale')
    parser.add_argument(
        '--aug_geom_tilt_max_deg', type=float, default=0.0, help='Augmentation: maximum rotation degree')
    parser.add_argument(
        '--aug_geom_wiggle_max_ratio', type=float, default=0.0,
        help='Augmentation: perspective warping level between 0 and 1')
    parser.add_argument(
        '--aug_geom_reflect', type=str2bool, default=False, help='Augmentation: Random horizontal flips')

    parser.add_argument(
        '--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Type of optimizer')
    parser.add_argument(
        '--optimizer_lr', type=float, default=0.01, help='Learning rate at start of training')
    parser.add_argument(
        '--optimizer_momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument(
        '--optimizer_weight_decay', type=float, default=0.001, help='Optimizer weight decay')
    parser.add_argument(
        '--optimizer_float_16', type=str2bool, default=False, help='Optimizer to use float16 precision')
    
    # custom to train bigger models with bigger batch sizes
    parser.add_argument(
        '--accumulate_grad_batches', type=int, default=1, help='Accumulated gradients run K small batches of size N before doing a backward pass')

    parser.add_argument(
        '--lr_scheduler', type=str, default='poly', choices=['poly'], help='Type of learning rate scheduler')
    parser.add_argument(
        '--lr_scheduler_power', type=float, default=0.9, help='Poly learning rate power')

    parser.add_argument(
        '--dataset', type=str, default='miniscapes', choices=['miniscapes'], help='Dataset name')
    
    parser.add_argument(
        '--tasks', type=str, nargs='+', default=['semseg'], choices=['depth', 'semseg'], help='Tasks to train on')

    parser.add_argument(
        '--model_name', type=str, default='deeplabv3p',
        choices=['deeplabv3p', 'deeplabv3p_multitask', 'adaptive_depth', 'deeplabv3p_multitask_task5'],
        help='CNN architecture')
    parser.add_argument(
        '--model_encoder_name', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        help='CNN architecture encoder')
    parser.add_argument(
        '--pretrained', type=str2bool, default=False, help='Use ImageNet pretrained weights')
    
    parser.add_argument(
        '--loss_weight_semseg', type=float, default=0.5, help='Weight of semantic segmentation loss')
    parser.add_argument(
        '--loss_weight_depth', type=float, default=0.5, help='Weight of depth estimation loss')

    parser.add_argument(
        '--workers', type=int, default=16, help='Number of worker threads fetching training data')
    parser.add_argument(
        '--workers_validation', type=int, default=4, help='Number of worker threads fetching validation data')

    parser.add_argument(
        '--num_steps_visualization_first', type=int, default=100, help='Visualization: first time step')
    parser.add_argument(
        '--num_steps_visualization_interval', type=int, default=1000, help='Visualization: interval in steps')
    parser.add_argument(
        '--visualize_num_samples_in_batch', type=int, default=8, help='Visualization: max number of samples in batch')
    parser.add_argument(
        '--visualize_img_grid_width', type=int, default=8, help='Visualization: number of samples per row')
    parser.add_argument(
        '--observe_train_ids', type=json.loads, default='[0,100]', help='Visualization: train IDs')
    parser.add_argument(
        '--observe_valid_ids', type=json.loads, default='[0,100]', help='Visualization: validation IDs')

    cfg = parser.parse_args()

    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg


EXPERIMENT_INVARIANT_KEYS = (
    'log_dir',
    'dataset_root',
    'prepare_submission',
    'batch_size_validation',
    'workers',
    'workers_validation',
    'num_steps_visualization_first',
    'num_steps_visualization_interval',
    'visualize_num_samples_in_batch',
    'visualize_img_grid_width',
    'observe_train_ids',
    'observe_valid_ids',
)
