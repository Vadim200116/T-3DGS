import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms

class LinearSegmentationHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_features: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear(x)
        probs = self.sigmoid(logits)
        probs = probs.reshape(-1, 24, 24)
        return probs

    @staticmethod
    def interpolate(probs, width, height):
        probs = F.interpolate(
            probs.reshape(-1, 1, 24, 24), 
            size=(width, height),
            mode='bilinear',
            align_corners=False
        )
        return probs

class DinoFeatureExatractor:
    def __init__(self, model_name = 'dinov2_vits14'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        dino_model.eval()
        
        self.dino_model = dino_model.to(device)
        self.preprocess = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image):
        img_prep = self.preprocess(image)
        with torch.no_grad():
            features_dict = self.dino_model.forward_features(img_prep)
        features = features_dict['x_norm_patchtokens']
        return features.squeeze(0)

    def extract_last_4(self, image):
        img_prep = self.preprocess(image)
        
        with torch.no_grad():
            features_dict = self.dino_model.get_intermediate_layers(
                img_prep,
                n=4,  # get last 4 layers
                reshape=False
            )
            
        features_list = []
        for features in features_dict:
            features_list.append(self.dino_model.norm(features.squeeze(0)))

        return torch.cat(features_list, dim=-1)

def dilate_mask(x, iterations=1):
    x = x.unsqueeze(0).unsqueeze(0)

    for _ in range(iterations):
        dilated_mask = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    
    return dilated_mask.squeeze()
