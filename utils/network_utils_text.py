from models.sst2 import distilbert_sst2

def get_network_text(network, **kwargs):
    networks = {
        'distilbert_sst2': distilbert_sst2,
    }
    if network not in networks:
        raise KeyError(f'Text network {network} is not supported.')
    return networks[network](**kwargs)