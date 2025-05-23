from optimizers import (KFACOptimizer,EKFACOptimizer,DNGD,SGD_,Adam_)
import torch.optim as optim
import inspect
def get_optimizer(args,net):
    OPTIMIZER_REGISTRY = {
        'sgd':       optim.SGD,
        'adam_':     Adam_,
        'kfac':      KFACOptimizer,
        'ekfac':     EKFACOptimizer,
        'dngd':      DNGD,
        'sgd_':      SGD_,
    }
    optim_name = args.optimizer.lower()
    if optim_name not in OPTIMIZER_REGISTRY:
        raise NotImplementedError(f"Optimizer '{optim_name}' is not implemented.")
    optim_cls = OPTIMIZER_REGISTRY[optim_name]
    common_kwargs = vars(args)  
    optim_kwargs = {}
    sig = inspect.signature(optim_cls.__init__)
    for name, param in sig.parameters.items():
        if name in ['self', 'params', 'model']:
            continue
        if name in common_kwargs:
            optim_kwargs[name] = common_kwargs[name]
    if 'params' in sig.parameters:
        optimizer = optim_cls(net.parameters(), **optim_kwargs)
    elif 'model' in sig.parameters:
        optimizer = optim_cls(net, **optim_kwargs)
    else:
        raise ValueError(f"Optimizer '{optim_name}' has unknown constructor signature.")
    return optimizer,optim_name