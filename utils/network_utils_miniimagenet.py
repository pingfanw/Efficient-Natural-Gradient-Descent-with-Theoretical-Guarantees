from models.miniimagenet import (densenet,resnet,wrn,pyramidnet,vit_small,vit_base,vit_large,vit_huge,HFDeiT,HFSwin,HFViT)

def get_network_miniimagenet(network,**kwargs):
    networks = {
        'densenet': densenet,
        'resnet': resnet,
        'wrn': wrn,
        'pyramidnet': pyramidnet,
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
        'vit_huge': vit_huge,
        'pretrain_vit': HFViT,
        'pretrain_swin': HFSwin,
        'pretrain_deit': HFDeiT
    }
    return networks[network](**kwargs)

