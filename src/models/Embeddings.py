import torch
import torch.nn as nn

# from omegaconf import OmegaConf
# config_path = './configs/anydoor.yaml'
# config = OmegaConf.load(config_path)
# DINOv2_weight_path = config.model.params.cond_stage_config.weight

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, device="cuda", freeze=True):
        super().__init__()
        # dinov2 = hubconf.dinov2_vitg14() 
        # state_dict = torch.load(DINOv2_weight_path)
        # dinov2.load_state_dict(state_dict, strict=False)
        torch.hub.set_dir('/root/autodl-tmp/torch_hub')
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
        self.projector = nn.Linear(1024,1024)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image,list):
            image = torch.cat(image,0)

        image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        # features = self.model.forward_features(image)
        image = image.to(torch.float16)
        features = self.model.backbone.forward_features(image)

        tokens = features["x_norm_patchtokens"]
        image_features  = features["x_norm_clstoken"]
        image_features = image_features.unsqueeze(1)
        hint = torch.cat([image_features,tokens],1) # 8,257,1024
        hint = hint.to(torch.float32)
        # print(f"{hint.shape=}")
        hint = self.projector(hint)
        return hint

    def encode(self, image):
        return self(image)