from .optimizers import KFACOptimizer,KFACOptimizer_NonFull,EKFACOptimizer,EKFACOptimizer_NonFull,DNGD,SGD_,Adam_

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
    elif name == 'SGD_':
        return SGD_
    elif name == 'Adam_':
        return Adam_
    else:
        raise NotImplementedError