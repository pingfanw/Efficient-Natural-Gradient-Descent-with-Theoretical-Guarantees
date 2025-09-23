# ./model/tinyimagenet/deit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageClassification

class HFDeiT(nn.Module):
    """
    HuggingFace DeiT wrapper for Tiny-ImageNet (200 classes).
    """
    def __init__(self, model_name: str = "facebook/deit-base-distilled-patch16-224", num_classes: int = 200):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        if x.dim() == 4 and (x.shape[-1] != 224 or x.shape[-2] != 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.model(pixel_values=x).logits
