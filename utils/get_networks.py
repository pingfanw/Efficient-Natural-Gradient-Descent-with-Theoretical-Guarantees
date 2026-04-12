TEXT_DATASETS = {'sst2', 'gsm8k'}


def normalize_dataset_name(name):
    if not isinstance(name, str):
        raise TypeError(f'dataset name must be str, but got {type(name)}')
    dataset = name.strip().lower().replace('-', '').replace('_', '')
    aliases = {
        'fashionmnist': 'fashionmnist',
        'cifar10': 'cifar10',
        'cifar100': 'cifar100',
        'svhn': 'svhn',
        'imagenet': 'imagenet',
        'miniimagenet': 'miniimagenet',
        'tinyimagenet': 'tinyimagenet',
        'mnist': 'mnist',
        'sst2': 'sst2',
        'gsm8k': 'gsm8k',
    }
    if dataset not in aliases:
        raise ValueError(f'Unsupported dataset name: {name}')
    return aliases[dataset]


def get_network(args):
    dataset = normalize_dataset_name(args.dataset)

    nc = {
        'mnist': 10,
        'fashionmnist': 10,
        'cifar10': 10,
        'cifar100': 100,
        'svhn': 10,
        'imagenet': 1000,
        'miniimagenet': 100,
        'tinyimagenet': 200,
        'sst2': 2,
        'gsm8k': 1,
    }
    idm = {
        'mnist': 28,
        'fashionmnist': 28,
        'cifar10': 32,
        'cifar100': 32,
        'svhn': 32,
        'imagenet': 224,
        'miniimagenet': 84,
        'tinyimagenet': 64,
        'sst2': getattr(args, 'max_seq_length', 128),
        'gsm8k': getattr(args, 'max_seq_length', 512),
    }
    icn = {
        'mnist': 1,
        'fashionmnist': 1,
        'cifar10': 3,
        'cifar100': 3,
        'svhn': 3,
        'imagenet': 3,
        'miniimagenet': 3,
        'tinyimagenet': 3,
        'sst2': 1,
        'gsm8k': 1,
    }

    args.dataset = dataset
    args.outputs_dim = nc[dataset]
    args.input_dim = idm[dataset]
    args.in_channels = icn[dataset]

    net = None

    if dataset in {'mnist', 'fashionmnist'}:
        from utils.network_utils_mnist import get_network_mnist
        net = get_network_mnist(args.network)

    if dataset in {'cifar10', 'cifar100'}:
        from utils.network_utils_cifar import get_network_cifar
        if args.network.lower() == 'preresnet':
            net = get_network_cifar(args.network, depth=args.depth, num_classes=args.outputs_dim)
        elif args.network.lower() == 'pyramidnet':
            net = get_network_cifar(
                args.network,
                depth=args.depth,
                alpha=48,
                input_shape=(1, args.in_channels, args.input_dim, args.input_dim),
                num_classes=args.outputs_dim,
                base_channels=16,
                block_type='bottleneck',
            )
        elif args.network.lower() in ['vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            net = get_network_cifar(
                args.network,
                mlp_ratio=args.mlp_ratio,
                input_size=args.input_dim,
                patch_size=args.patch_size,
                in_channels=args.in_channels,
                num_classes=args.outputs_dim,
            )
        else:
            net = get_network_cifar(
                args.network,
                depth=args.depth,
                num_classes=args.outputs_dim,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                widen_factor=args.widen_factor,
                dropRate=args.dropRate,
            )

    if dataset == 'svhn':
        from utils.network_utils_svhn import get_network_svhn
        if args.network.lower() == 'preresnet':
            net = get_network_svhn(args.network, depth=args.depth, num_classes=args.outputs_dim)
        elif args.network.lower() == 'pyramidnet':
            net = get_network_svhn(
                args.network,
                depth=args.depth,
                alpha=48,
                input_shape=(1, args.in_channels, args.input_dim, args.input_dim),
                num_classes=args.outputs_dim,
                base_channels=16,
                block_type='bottleneck',
            )
        elif args.network.lower() in ['vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            net = get_network_svhn(
                args.network,
                mlp_ratio=args.mlp_ratio,
                input_size=args.input_dim,
                patch_size=args.patch_size,
                in_channels=args.in_channels,
                num_classes=args.outputs_dim,
            )
        else:
            net = get_network_svhn(
                args.network,
                depth=args.depth,
                num_classes=args.outputs_dim,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                widen_factor=args.widen_factor,
                dropRate=args.dropRate,
            )

    if dataset == 'imagenet':
        from utils.network_utils_imagenet import get_network_imagenet
        if args.network.lower() == 'pyramidnet':
            net = get_network_imagenet(
                args.network,
                depth=args.depth,
                alpha=48,
                input_shape=(1, args.in_channels, args.input_dim, args.input_dim),
                num_classes=args.outputs_dim,
                base_channels=16,
                block_type='bottleneck',
            )
        elif args.network.lower() in ['vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            net = get_network_imagenet(
                args.network,
                mlp_ratio=args.mlp_ratio,
                input_size=args.input_dim,
                patch_size=args.patch_size,
                in_channels=args.in_channels,
                num_classes=args.outputs_dim,
            )
        elif args.network.lower() in ['pretrain_vit', 'pretrain_swin', 'pretrain_deit', 'pretrain_beit', 'pretrain_pvt']:
            net = get_network_imagenet(args.network, num_classes=args.outputs_dim)
        else:
            net = get_network_imagenet(
                args.network,
                depth=args.depth,
                num_classes=args.outputs_dim,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                widen_factor=args.widen_factor,
                dropRate=args.dropRate,
            )

    if dataset == 'miniimagenet':
        from utils.network_utils_miniimagenet import get_network_miniimagenet
        if args.network.lower() == 'pyramidnet':
            net = get_network_miniimagenet(
                args.network,
                depth=args.depth,
                alpha=48,
                input_shape=(1, args.in_channels, args.input_dim, args.input_dim),
                num_classes=args.outputs_dim,
                base_channels=16,
                block_type='bottleneck',
            )
        elif args.network.lower() in ['vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            net = get_network_miniimagenet(
                args.network,
                mlp_ratio=args.mlp_ratio,
                input_size=args.input_dim,
                patch_size=args.patch_size,
                in_channels=args.in_channels,
                num_classes=args.outputs_dim,
            )
        elif args.network.lower() in ['pretrain_vit', 'pretrain_swin', 'pretrain_deit', 'pretrain_beit', 'pretrain_pvt']:
            net = get_network_miniimagenet(args.network, num_classes=args.outputs_dim)
        else:
            net = get_network_miniimagenet(
                args.network,
                depth=args.depth,
                num_classes=args.outputs_dim,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                widen_factor=args.widen_factor,
                dropRate=args.dropRate,
            )

    if dataset == 'tinyimagenet':
        from utils.network_utils_tinyimagenet import get_network_tinyimagenet
        if args.network.lower() == 'pyramidnet':
            net = get_network_tinyimagenet(
                args.network,
                depth=args.depth,
                alpha=48,
                input_shape=(1, args.in_channels, args.input_dim, args.input_dim),
                num_classes=args.outputs_dim,
                base_channels=16,
                block_type='bottleneck',
            )
        elif args.network.lower() in ['vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            net = get_network_tinyimagenet(
                args.network,
                mlp_ratio=args.mlp_ratio,
                input_size=args.input_dim,
                patch_size=args.patch_size,
                in_channels=args.in_channels,
                num_classes=args.outputs_dim,
            )
        elif args.network.lower() in ['pretrain_vit', 'pretrain_swin', 'pretrain_deit', 'pretrain_beit', 'pretrain_pvt']:
            net = get_network_tinyimagenet(args.network, num_classes=args.outputs_dim)
        else:
            net = get_network_tinyimagenet(
                args.network,
                depth=args.depth,
                num_classes=args.outputs_dim,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                widen_factor=args.widen_factor,
                dropRate=args.dropRate,
            )

    if dataset in TEXT_DATASETS:
        from utils.network_utils_text import get_network_text
        net = get_network_text(
            args.network,
            model_name=args.text_model_name,
            num_classes=args.outputs_dim,
            cache_dir=args.model_path,
        )

    if net is None:
        raise ValueError(f'Network {args.network} not supported for dataset {args.dataset}')

    net = net.to(args.device)
    return net