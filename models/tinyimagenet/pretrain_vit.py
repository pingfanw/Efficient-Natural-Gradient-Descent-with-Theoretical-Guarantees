import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageClassification

class HFViT(nn.Module):
    """
    HuggingFace ViT wrapper for Tiny-ImageNet (200 classes).
    Auto-resizes input to 224x224 to keep main loop & transforms untouched.
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224", num_classes: int = 200):
        
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: [B, C, H, W], auto-upsample to 224
        if x.dim() == 4 and (x.shape[-1] != 224 or x.shape[-2] != 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out = self.model(pixel_values=x)
        return out.logits
