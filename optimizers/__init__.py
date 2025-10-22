from .optimizers import KFACOptimizer,EKFACOptimizer,DNGD,SGD_,Adam_,AdaGrad_,AdamW,Muon

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
    elif name == 'AdaGrad_':
        return AdaGrad_
    elif name == 'AdamW':
        return AdamW
    elif name == 'muon':
        return Muon,
    else:
        raise NotImplementedError