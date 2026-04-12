from models.gsm8k import qwen3_0_6b, qwen3_1_7b, qwen3_8b
from models.sst2 import distilbert_sst2


def get_network_text(network, **kwargs):
    networks = {
        'distilbert_sst2': distilbert_sst2,
        'qwen3_0_6b': qwen3_0_6b,
        'qwen3_1_7b': qwen3_1_7b,
        'qwen3_8b': qwen3_8b,
    }
    if network not in networks:
        raise KeyError(f'Text network {network} is not supported.')
    return networks[network](**kwargs)
