from models.imagenet import (densenet, resnet, wrn, pyramidnet)

def get_network_imagenet(network, **kwargs):
    networks = {
        'densenet': densenet,
        'resnet': resnet,
        'wrn': wrn,
        'pyramidnet': pyramidnet
    }
    return networks[network](**kwargs)

