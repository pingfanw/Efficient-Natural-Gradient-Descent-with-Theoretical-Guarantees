import torch.optim as optim

from optimizers import AdaGrad_, AdamW, Adam_, DNGD, EKFACOptimizer, KFACOptimizer, Muon, NGD, SAM, SGD_

OPTIMIZER_REGISTRY = {
    'sgd': optim.SGD,
    'adam_': Adam_,
    'kfac': KFACOptimizer,
    'ekfac': EKFACOptimizer,
    'dngd': DNGD,
    'sgd_': SGD_,
    'adagrad_': AdaGrad_,
    'adamw': AdamW,
    'muon': Muon,
    'ngd': NGD,
    'sam': SAM,
}

SAM_BASE_OPTIMIZER_REGISTRY = {
    'sgd': optim.SGD,
    'sgd_': SGD_,
    'adam_': Adam_,
    'adagrad_': AdaGrad_,
    'adamw': AdamW,
}

OPTIMIZER_SWEEP_PARAMS = {
    'sgd': ['learning_rate', 'momentum', 'weight_decay'],
    'sgd_': ['learning_rate', 'momentum', 'weight_decay'],
    'adam_': ['learning_rate', 'betas', 'eps', 'weight_decay', 'amsgrad'],
    'adagrad_': ['learning_rate', 'eps', 'weight_decay'],
    'adamw': ['learning_rate', 'betas', 'eps', 'weight_decay'],
    'muon': ['learning_rate', 'betas', 'eps', 'weight_decay', 'retraction_eps'],
    'dngd': ['learning_rate', 'momentum', 'weight_decay', 'damping'],
    'ngd': ['learning_rate', 'momentum', 'weight_decay', 'damping'],
    'kfac': ['learning_rate', 'momentum', 'weight_decay', 'kl_clip', 'damping', 'stat_decay', 'TCov', 'TInv'],
    'ekfac': ['learning_rate', 'momentum', 'weight_decay', 'kl_clip', 'damping', 'stat_decay', 'TCov', 'TInv', 'TScal'],
    'sam': ['learning_rate', 'momentum', 'weight_decay', 'rho', 'sam_base_optimizer', 'betas', 'eps', 'amsgrad'],
}


def get_supported_optimizer_params(optimizer_name):
    optimizer_name = optimizer_name.lower()
    if optimizer_name not in OPTIMIZER_SWEEP_PARAMS:
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented.")
    return OPTIMIZER_SWEEP_PARAMS[optimizer_name]


def _build_optimizer_kwargs(args, optim_name):
    if optim_name in {'sgd', 'sgd_'}:
        return dict(lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    if optim_name == 'adam_':
        return dict(lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay, eps=args.eps, amsgrad=args.amsgrad)
    if optim_name == 'adagrad_':
        return dict(lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.eps)
    if optim_name == 'adamw':
        return dict(lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay, eps=args.eps)
    if optim_name == 'muon':
        return dict(lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay, eps=args.eps,
                    retraction_eps=args.retraction_eps)
    if optim_name in {'dngd', 'ngd'}:
        return dict(lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, damping=args.damping)
    if optim_name == 'kfac':
        return dict(lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                    kl_clip=args.kl_clip, damping=args.damping, stat_decay=args.stat_decay, TCov=args.TCov, TInv=args.TInv)
    if optim_name == 'ekfac':
        return dict(lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                    kl_clip=args.kl_clip, damping=args.damping, stat_decay=args.stat_decay,
                    TCov=args.TCov, TInv=args.TInv, TScal=args.TScal)
    if optim_name == 'sam':
        return dict(rho=args.rho)
    raise ValueError(f"Optimizer '{optim_name}' has unknown constructor signature.")


def get_optimizer(args, net):
    optim_name = args.optimizer.lower()
    if optim_name not in OPTIMIZER_REGISTRY:
        raise NotImplementedError(f"Optimizer '{optim_name}' is not implemented.")

    optim_cls = OPTIMIZER_REGISTRY[optim_name]
    optim_kwargs = _build_optimizer_kwargs(args, optim_name)

    if optim_name == 'sam':
        base_name = args.sam_base_optimizer.lower()
        if base_name not in SAM_BASE_OPTIMIZER_REGISTRY:
            raise ValueError(
                f"SAM base optimizer '{base_name}' is not supported. Supported base optimizers: {sorted(SAM_BASE_OPTIMIZER_REGISTRY)}"
            )
        base_kwargs = _build_optimizer_kwargs(args, base_name)
        optimizer = optim_cls(
            net.parameters(),
            base_optimizer=SAM_BASE_OPTIMIZER_REGISTRY[base_name],
            **optim_kwargs,
            **base_kwargs,
        )
        return optimizer, f'{optim_name}_{base_name}'

    if optim_name in {'dngd', 'kfac', 'ekfac', 'ngd'}:
        optimizer = optim_cls(net, **optim_kwargs)
    else:
        optimizer = optim_cls(net.parameters(), **optim_kwargs)

    return optimizer, optim_name
