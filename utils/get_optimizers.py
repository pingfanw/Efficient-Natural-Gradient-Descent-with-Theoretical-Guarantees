from optimizers import (KFACOptimizer,EKFACOptimizer,DNGD,SGD_,Adam_,AdaGrad_,AdamW,Muon)
import torch.optim as optim
def get_optimizer(args,net):
    OPTIMIZER_REGISTRY = {
        'sgd':       optim.SGD,
        'adam_':     Adam_,
        'kfac':      KFACOptimizer,
        'ekfac':     EKFACOptimizer,
        'dngd':      DNGD,
        'sgd_':      SGD_,
        'adagrad_':  AdaGrad_,
        'adamw':     AdamW,
        'muon':      Muon,
    }
    optim_name = args.optimizer.lower()
    if optim_name not in OPTIMIZER_REGISTRY:
        raise NotImplementedError(f"Optimizer '{optim_name}' is not implemented.")
    optim_cls = OPTIMIZER_REGISTRY[optim_name]
    if optim_name == 'sgd':
        optimizer = optim_cls(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif optim_name == 'adam_':
        optimizer = optim_cls(net.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay, eps=args.eps, amsgrad=args.amsgrad)
    elif optim_name == 'sgd_':
        optimizer = optim_cls(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif optim_name == 'adagrad_':
        optimizer = optim_cls(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.eps)
    elif optim_name == 'adamw':
        optimizer = optim_cls(net.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay, eps=args.eps)
    elif optim_name == 'muon':
        optimizer = optim_cls(net.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay, eps=args.eps, retraction_eps=args.retraction_eps)
    elif optim_name == 'dngd':
        optimizer = optim_cls(net, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, damping=args.damping)
    elif optim_name == 'kfac':
        optimizer = optim_cls(net, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                            kl_clip=args.kl_clip, 
                            damping=args.damping, 
                            stat_decay=args.stat_decay, 
                            TCov=args.TCov,
                            TInv=args.TInv)
    elif optim_name == 'ekfac':
        optimizer = optim_cls(net, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                            kl_clip=args.kl_clip, 
                            damping=args.damping, 
                            stat_decay=args.stat_decay, 
                            TCov=args.TCov,
                            TInv=args.TInv,
                            TScal=args.TScal)
    else:
        raise ValueError(f"Optimizer '{optim_name}' has unknown constructor signature.")
    return optimizer,optim_name