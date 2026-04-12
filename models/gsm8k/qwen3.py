import torch.nn as nn


DEFAULT_QWEN3_MODEL_NAMES = {
    'qwen3_0_6b': 'Qwen/Qwen3-0.6B-Base',
    'qwen3_1_7b': 'Qwen/Qwen3-1.7B-Base',
    'qwen3_8b': 'Qwen/Qwen3-8B-Base',
}


class Qwen3Model(nn.Module):
    def __init__(self, model_name='Qwen/Qwen3-0.6B-Base', cache_dir=None, **kwargs):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                'transformers is required for GSM8K/Qwen3 support. Please install transformers>=4.51.0 before using dataset=gsm8k.'
            ) from exc

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        if self.backbone.config.pad_token_id is None and self.backbone.config.eos_token_id is not None:
            self.backbone.config.pad_token_id = self.backbone.config.eos_token_id

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)


def qwen3_0_6b(**kwargs):
    kwargs.setdefault('model_name', DEFAULT_QWEN3_MODEL_NAMES['qwen3_0_6b'])
    return Qwen3Model(**kwargs)


def qwen3_1_7b(**kwargs):
    kwargs.setdefault('model_name', DEFAULT_QWEN3_MODEL_NAMES['qwen3_1_7b'])
    return Qwen3Model(**kwargs)


def qwen3_8b(**kwargs):
    kwargs.setdefault('model_name', DEFAULT_QWEN3_MODEL_NAMES['qwen3_8b'])
    return Qwen3Model(**kwargs)
