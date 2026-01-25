import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageClassification


class HFPvt(nn.Module):
    """
    HuggingFace PVT wrapper for image classification (Mini-ImageNet by default).

    Typical checkpoints (HuggingFace Hub):
        - "Zetatech/pvt-tiny-224"
        - "Zetatech/pvt-small-224"
        - "Zetatech/pvt-medium-224"
        - "Zetatech/pvt-large-224"

    You can pass any PVT image-classification checkpoint name here.
    """
    def __init__(
        self,
        model_name: str = "Zetatech/pvt-small-224",
        num_classes: int = 100,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        # Expect x: [B, C, H, W], auto-upsample to 224×224
        if x.dim() == 4 and (x.shape[-1] != 224 or x.shape[-2] != 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.model(pixel_values=x).logits
