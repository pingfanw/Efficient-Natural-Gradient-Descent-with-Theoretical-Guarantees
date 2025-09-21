from utils.network_utils_cifar import get_network_cifar
from utils.network_utils_mnist import get_network_mnist
from utils.network_utils_imagenet import get_network_imagenet
from utils.network_utils_miniimagenet import get_network_miniimagenet
from utils.network_utils_svhn import get_network_svhn
from utils.network_utils_tinyimagenet import get_network_tinyimagenet

def get_network(args):
    nc = { 
        'mnist': 10,
        'fashionmnist': 10,
        'cifar10': 10,
        'cifar100': 100,
        'svhn': 10,
        'imagenet': 1000,
        'miniimagenet': 100,
        'tinyimagenet': 200
    }
    idm = {
        'mnist': 28,
        'fashionmnist': 28,
        'cifar10': 32,
        'cifar100': 32,
        'svhn': 32,
        'imagenet': 224,
        'miniimagenet': 84,
        'tinyimagenet': 64
    }
    icn = {
        'mnist': 1,
        'fashionmnist': 1,
        'cifar10': 3,
        'cifar100': 3,
        'svhn': 3,
        'imagenet': 3,
        'miniimagenet': 3,
        'tinyimagenet': 3
    }
    args.outputs_dim = nc[args.dataset.lower()]
    args.input_dim = idm[args.dataset.lower()]
    args.in_channels = icn[args.dataset.lower()]
    net = None
    if args.dataset.lower() == 'mnist' or args.dataset.lower() == 'fashionmnist':
            net = get_network_mnist(args.network)
    if args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100':
        if args.network.lower() == 'preresnet':
            net = get_network_cifar(args.network,depth=args.depth,num_classes=args.outputs_dim)
        elif args.network.lower() == 'pyramidnet':
            net = get_network_cifar(args.network,depth = args.depth,alpha = 48,input_shape=(1,args.in_channels,args.input_dim,args.input_dim),num_classes = args.outputs_dim,base_channels = 16,block_type = 'bottleneck')
        elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
            net = get_network_cifar(args.network,mlp_ratio=args.mlp_ratio,input_size=args.input_dim,patch_size=args.patch_size,in_channels=args.in_channels,num_classes=args.outputs_dim)
        else:
            net = get_network_cifar(args.network,depth=args.depth,num_classes=args.outputs_dim,growthRate=args.growthRate,compressionRate=args.compressionRate,widen_factor=args.widen_factor,dropRate=args.dropRate)
    if args.dataset.lower() == 'svhn':
        if args.network.lower() == 'preresnet':
            net = get_network_svhn(args.network,depth=args.depth,num_classes=args.outputs_dim)
        elif args.network.lower() == 'pyramidnet':
            net = get_network_svhn(args.network,depth = args.depth,alpha = 48,input_shape=(1,args.in_channels,args.input_dim,args.input_dim),num_classes = args.outputs_dim,base_channels = 16,block_type = 'bottleneck')
        elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
            net = get_network_svhn(args.network,mlp_ratio=args.mlp_ratio,input_size=args.input_dim,patch_size=args.patch_size,in_channels=args.in_channels,num_classes=args.outputs_dim)
        else:
            net = get_network_svhn(args.network,depth=args.depth,num_classes=args.outputs_dim,growthRate=args.growthRate,compressionRate=args.compressionRate,widen_factor=args.widen_factor,dropRate=args.dropRate)
    if args.dataset.lower() == 'imagenet':
        if args.network.lower() == 'pyramidnet':
            net = get_network_imagenet(args.network,depth=args.depth,alpha=48,input_shape=(1,args.in_channels,args.input_dim,args.input_dim),num_classes=args.outputs_dim,base_channels=16,block_type='bottleneck')
        elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
            net = get_network_imagenet(args.network,mlp_ratio=args.mlp_ratio,input_size=args.input_dim,patch_size=args.patch_size,in_channels=args.in_channels,num_classes=args.outputs_dim)
        else:
            net = get_network_imagenet(args.network,depth=args.depth,num_classes=args.outputs_dim,growthRate=args.growthRate,compressionRate=args.compressionRate,widen_factor=args.widen_factor,dropRate=args.dropRate)
    if args.dataset.lower() == 'miniimagenet':
        if args.network.lower() == 'pyramidnet':
            net = get_network_miniimagenet(args.network,depth=args.depth,alpha=48,input_shape=(1,args.in_channels,args.input_dim,args.input_dim),num_classes=args.outputs_dim,base_channels=16,block_type='bottleneck')
        elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
            net = get_network_miniimagenet(args.network,mlp_ratio=args.mlp_ratio,input_size=args.input_dim,patch_size=args.patch_size,in_channels=args.in_channels,num_classes=args.outputs_dim)
        else:
            net = get_network_miniimagenet(args.network,depth=args.depth,num_classes=args.outputs_dim,growthRate=args.growthRate,compressionRate=args.compressionRate,widen_factor=args.widen_factor,dropRate=args.dropRate)
    if args.dataset.lower() == "tinyimagenet":
        if args.network.lower() == 'pyramidnet':
            net = get_network_tinyimagenet(args.network,depth=args.depth,alpha=48,input_shape=(1,args.in_channels,args.input_dim,args.input_dim),num_classes=args.outputs_dim,base_channels=16,block_type='bottleneck')
        elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
            net = get_network_tinyimagenet(args.network,mlp_ratio=args.mlp_ratio,input_size=args.input_dim,patch_size=args.patch_size,in_channels=args.in_channels,num_classes=args.outputs_dim)
        else:
            net = get_network_tinyimagenet(args.network,depth=args.depth,num_classes=args.outputs_dim,growthRate=args.growthRate,compressionRate=args.compressionRate,widen_factor=args.widen_factor,dropRate=args.dropRate)
    if net is None:
        raise ValueError(f"Network {args.network} not supported for dataset {args.dataset}")
    net = net.to(args.device)
    return net