import argparse


def params_init():
    parser = argparse.ArgumentParser(description='Data recovery in VFL')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'Bank'])
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--save', type=bool, default=False, help='whether to save every model')

    #-------------- -----------VFL----------------------------
    parser.add_argument('--vfl_model', type=str, default='resnet20', help='the bottom model type of in VFL training',
                        choices=['resnet8', 'resnet14', 'resnet20'])
    parser.add_argument('--vfl_epochs', type=int, default=50, help='number of total epochs to run')
    parser.add_argument('--vfl_batch_size', type=int, default=32, help='mini-batch size (default: 128)')
    parser.add_argument('--vfl_lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--vfl_momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--vfl_weight_decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--vfl_step_gamma', type=float, default=0.1, help='gamma for step scheduler')

    #-------------------------MAE----------------------------
    # Image Features -- As the official MAE is adopted, they are uniformly set to 224/16
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size')

    # model training
    parser.add_argument('--mae_batch_size', type=int, default=16, help='batch size for MAE pre-training')
    parser.add_argument('--mae_pretrain_epochs', type=int, default=200, help='number of total epochs to run: cifar(200) and tiny-imagenet(500)')
    parser.add_argument('--mae_finetune_epochs', type=int, default=200, help='number of total epochs to run: cifar(200) and tiny-imagenet(500)')
    parser.add_argument('--mae_warm_epochs', type=int, default=6, help='number of epochs for warm-up lr-schedule')
    parser.add_argument('--mae_lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--mae_warm_start_lr', type=float, default=1e-6, help='warm-up start learning rate')
    parser.add_argument('--mae_warm_end_lr', type=float, default=1e-7, help='warm-up end learning rate')

    #-------------------------Recovery----------------------------
    parser.add_argument('--is_recovery_supervised', type=bool, default=False,
                        help='whether to train supervised attack model. True(supervised training) or False(unsupervised)')

    # supervised training
    parser.add_argument('--attack_batch_size', type=int, default=16, help='batch size for data recovery')
    parser.add_argument('--attack_epochs', type=int, default=100, help='number of total epochs to run')
    parser.add_argument('--attack_lr', type=int, default=0.01, help='learning rate')
    parser.add_argument('--attack_noise_type', type=str, default='None', help='type of attack noise', choices=['None', 'Noise', 'Soteria'])

    args = parser.parse_known_args()[0]

    return args

