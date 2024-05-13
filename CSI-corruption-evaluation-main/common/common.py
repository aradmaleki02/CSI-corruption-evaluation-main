from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--dataset', help='Dataset',
                        choices=['imagenet30', 'fmnist', 'mnist', 'fashion-mnist', 'svhn-10-corruption', 'svhn-10', 'cifar100-corruption', 'svhn', 'cifar10-corruption', 'cifar10', 'cifar100', 'imagenet'], default="cifar10", type=str)
    parser.add_argument('--desired_attack', help='desired_attack',
                        choices=['PGD', 'FGSM'],
                        default="PGD", type=str)
    parser.add_argument("--in_attack", help='save ood score for plotting histogram',
                        default=False, action='store_true')
    parser.add_argument("--out_attack", help='save ood score for plotting histogram',
                        default=False, action='store_true')
    parser.add_argument('--eps', type=float, default=0.0314,
                        help='maximum perturbation of adversaries (8/255 for cifar-10)')
    
    parser.add_argument('--steps', type=int, default=10,
                        help='maximum iteration when generating adversarial examples')

    parser.add_argument('--noise_mean', help='',
                        default=0.0, type=float)
    parser.add_argument('--noise_std', help='',
                        default=1.0, type=float)
    parser.add_argument('--noise_scale', help='',
                        default=0.031372549, type=float)

    parser.add_argument('--one_class_idx', help='None: multi-class, Not None: one-class',
                        default=None, type=int)
    parser.add_argument('--cifar_corruption_data', help='',
                        default="./CIFAR-10-C/defocus_blur.npy", type=str)
    parser.add_argument('--model', help='Model',
                        choices=['resnet18', 'resnet18_imagenet'], default="resnet18", type=str)
    parser.add_argument('--mode', help='Training mode',
                        default='ood_pre', type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='none',
                        choices=['rotation', 'cutperm', 'none'], type=str)

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default="./cifar10_oc_class0.model", type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=5, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=10, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)

    ##### Objective Configurations #####
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                        default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',
                        default=0.5, type=float)

    ##### Evaluation Configurations #####
    parser.add_argument("--ood_dataset", help='Datasets for OOD detection',
                        default=None, nargs="*", type=str)
    parser.add_argument("--ood_score", help='score function for OOD detection',
                        default=['norm_mean'], nargs="+", type=str)
    parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices=['penultimate', 'simclr', 'shift'],
                        default=['simclr', 'shift'], nargs="+", type=str)
    parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default=1, type=int)
    parser.add_argument("--ood_batch_size", help='batch size to compute OOD score',
                        default=100, type=int)
    parser.add_argument("--test_id", help='batch size to compute OOD score',
                        default=1, type=int)
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                        default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                        action='store_true')

    parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
    parser.add_argument("--save_score", help='save ood score for plotting histogram',
                        action='store_true')

    if default:
        return parser.parse_args('')  # empty string
    else:
        # return parser.parse_args()
        args, unknown = parser.parse_known_args()
        return args