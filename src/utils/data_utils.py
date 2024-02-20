import torch
from einops import rearrange, repeat


def mask_features(features: list, mask: torch.Tensor, is_5d=True):
    """
    Mask features with the given mask.
    """
    if is_5d:
        mask = rearrange(mask, 'b f c h w -> (b f) c h w')

    for i, feature in enumerate(features):
        # Resize the mask to the feature size.
        mask = torch.nn.functional.interpolate(mask, size=feature.shape[-2:])

        # Mask the feature.
        features[i] = feature * (1 - mask)

    return features
