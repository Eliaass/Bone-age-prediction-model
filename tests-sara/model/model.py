from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch
import timm

class BAA_resnet18(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained resnet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        last_width = self.backbone.fc.in_features

        # make it take grayscale instead of RGB: need to replace first conv layer since i turn imgs to grayscale
        c = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = c.out_channels,
            kernel_size = c.kernel_size, 
            stride = c.stride, 
            padding = c.padding, 
            bias = c.bias
        )

        # replace classifier with regression head: replace backbone.fc with useless layer and store fc elsewhere (for adding gender input to reg head later)
        self.backbone.fc = nn.Identity()
        self.regression_head = nn.Linear(last_width + 1, 1)


    def forward(self, img_tensor, gender_tensor):
        features = self.backbone(img_tensor)
        gender_tensor = gender_tensor.unsqueeze(1)

        fc_input = torch.cat([features, gender_tensor], dim = 1)
        out = self.regression_head(fc_input)

        return out.squeeze(1)


class BoneAgeEfficientNet(nn.Module):
    def __init__(self, model_name, pretrained=True, drop_path_rate=0.1, head_dropout=0.2, hidden_dim=1024):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg", drop_path_rate=drop_path_rate)
        feature_dim = self.backbone.num_features
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim), nn.SiLU(), nn.Dropout(head_dropout), nn.Linear(hidden_dim, 1)
        )
    def forward(self, image, male):
        features = self.backbone(image)

        male = male.unsqueeze(1)

        x = torch.cat([features, male], dim=1)

        return self.regression_head(x).squeeze(1)
