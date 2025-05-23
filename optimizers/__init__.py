from .optimizers import KFACOptimizer,EKFACOptimizer,DNGD,SGD_,Adam_

def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'ekfac':
        return EKFACOptimizer
    elif name == 'dngd':
        return DNGD
    elif name == 'SGD_':
        return SGD_
    elif name == 'Adam_':
        return Adam_
    else:
        raise NotImplementedError