import torch.nn as nn

class DistilBertSST2Classifier(nn.Module):
    def __init__(self, model_name='distilbert/distilbert-base-uncased', num_classes=2,
                 cache_dir=None):
        super().__init__()
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:
            raise ImportError(
                'transformers is required for SST-2 support. Please install transformers before using dataset=sst2.'
            ) from exc

        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            cache_dir=cache_dir,
        )

    def forward(self, **inputs):
        return self.backbone(**inputs)


def distilbert_sst2(**kwargs):
    return DistilBertSST2Classifier(**kwargs)