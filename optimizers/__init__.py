from .optimizers import KFACOptimizer,KFACOptimizer_NonFull,EKFACOptimizer,EKFACOptimizer_NonFull,DNGD,SGD_mod,Adam_mod

def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'kfac_nonfull':
        return KFACOptimizer_NonFull
    elif name == 'ekfac':
        return EKFACOptimizer
    elif name == 'ekfac_nonfull':
        return EKFACOptimizer_NonFull
    elif name == 'dngd':
        return DNGD
    elif name == 'sgd_mod':
        return SGD_mod
    elif name == 'adam_mod':
        return Adam_mod
    else:
        raise NotImplementedError