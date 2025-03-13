import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
from featup.util import norm, unnorm

class FeatUp_FeatureExtractor:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(device)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            norm
        ])
    
    def extract(self, images):
        image_prep = self.preprocess(images)
        with torch.no_grad():
            features = self.upsampler(image_prep)
        return features


def dilate_mask(x, iterations=1):
    dilated_mask = x.unsqueeze(0).unsqueeze(0)

    for _ in range(iterations):
        dilated_mask = F.max_pool2d(dilated_mask, kernel_size=3, stride=1, padding=1)
    
    return dilated_mask.squeeze()


def pad_and_unpad(func):
    def wrapper(self, image, feature_extractor):
        width, height = image.shape[2:]
        target_width = (width + 13) // 14 * 14
        target_height = (height + 13) // 14 * 14

        pad_width_left = (target_width - width) // 2
        pad_width_right = target_width - width - pad_width_left
        pad_height_top = (target_height - height) // 2
        pad_height_bottom = target_height - height - pad_height_top

        padded_input = F.pad(image, 
                            (pad_height_top, pad_height_bottom, pad_width_left, pad_width_right), 
                            mode='replicate')

        weights = func(self, padded_input, feature_extractor).squeeze()
        rec_weights = weights[:,:,pad_width_left: pad_width_left + width, pad_height_top: pad_height_top + height]

        return rec_weights

    return wrapper

class DinoFeatureExatractor:
    def __init__(self, model_name = 'dinov2_vits14'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        dino_model.eval()
        
        self.dino_model = dino_model.to(device)
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image):
        img_prep = self.preprocess(image)
        with torch.no_grad():
            features_dict = self.dino_model.forward_features(img_prep)
        features = features_dict['x_norm_patchtokens']
        return features

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

class DPT_Head(nn.Module):
    def __init__(self, features):
        super(DPT_Head, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features, features // 3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, features // 3),
            nn.GELU(), 
            nn.Conv2d(features // 3, features // 3, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 3, features // 6, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, features // 6),
            nn.GELU(),
            nn.Conv2d(features // 6, features // 6, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 6, features // 12, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, features // 12),
            #nn.ReLU(),
            #nn.Conv2d(features // 12, features // 12, kernel_size=3, stride=1, padding=1),

            #
            #Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            #nn.Conv2d(features // 12, features // 12, kernel_size=3, stride=1, padding=1),
            #nn.GroupNorm(1, features // 12),
            
            #nn.ReLU(),
            #nn.Conv2d(features // 12, features // 12, kernel_size=1, stride=1, padding=0),
            #

            nn.GELU(),
            nn.Conv2d(features // 12, 3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ConvDPT(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.head = DPT_Head(self.feature_dim)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    @pad_and_unpad
    def forward(self, image, feature_extractor):
        width, height = image.shape[2:]
        features = feature_extractor.extract(image)
        features = features.reshape(-1, width//14, height//14, self.feature_dim).permute(0, 3, 1, 2)
        logits = self.head(features)
        logits = F.interpolate(
            logits, 
            size=(width, height),
            mode='bilinear',
            align_corners=False
        )
        sigma = self.softplus(logits[:, :2])
        rho = self.tanh(logits[:, 2:3])
        covariations = torch.cat([sigma, rho], dim=1)
        return covariations