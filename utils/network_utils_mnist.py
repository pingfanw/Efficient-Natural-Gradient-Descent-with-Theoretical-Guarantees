from models.mnist import (mlp)

def get_network_mnist(network, **kwargs):
    networks = {
        'mlp': mlp
    }
    return networks[network](**kwargs)

