import yaml
import os
from argparse import ArgumentParser

def load_cfg(args):
    with open(args.config, "rb") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        args.__dict__[key] = value

    return args

def parse_args():
    parser = ArgumentParser()

    """ Config """
    parser.add_argument('--config', help='Path to config .yaml file', type=str, default=None)
    parser.add_argument('--seed', help='Random seed', type=int, default=42)

    """ Data """
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--metadata_csv', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=256)


    """ Model """
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--p_dropout', type=float, default=0.3)

    """ Optimizer """
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler', type=str, default=None)

    """ Criterion """
    parser.add_argument('--criterion', type=str, default='bce')
    parser.add_argument('--mixup_alpha', type=float, default=0.0)

    parser.add_argument('--masking', type=str, default='none',
                        choices=['none', 'random_box', 'random_patch', 'slice'],
                        help='Masking strategy to use during training')
    parser.add_argument('--mask_prob', type=float, default=0.5,
                        help='Probability of applying masking to each batch (0.0 to 1.0)')

    parser.add_argument('--num_masks', type=int, default=2,
                        help='Number of random boxes to mask (for random_box strategy)')
    parser.add_argument('--mask_min_size', type=float, default=0.2,
                        help='Minimum size of mask boxes as fraction of volume dimension')
    parser.add_argument('--mask_max_size', type=float, default=0.4,
                        help='Maximum size of mask boxes as fraction of volume dimension')
    parser.add_argument('--mask_ratio', type=float, default=0.4,
                        help='Fraction of patches/slices to mask')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[32, 32, 16],
                        help='Patch size for random_patch masking (D H W)')

    # Anatomical masking arguments
    parser.add_argument('--num_regions_to_mask', type=int, default=3,
                        help='Number of anatomical octants to mask (1-7)')
    """ Hyperparameter """
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    """ Training """
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_best', type=str, default='accuracy')
    parser.add_argument('--early_stopping_patience', type=int, default=0)

    """ Logging """
    parser.add_argument('--log_tensorboard', type=eval, default=True)
    parser.add_argument('--log_interval', type=int, default=100)

    """ Parse arguments """
    args = parser.parse_args()

    """ Load config file """
    if args.config is not None and os.path.exists(args.config):
        load_cfg(args)

    return args
