from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch

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

        # specify the target layer for gradcam inside the model class to simplify testing different models
        # target layer needs to be the last convolutional layer: for resnet18, final block of layers is 4, n we want last layer only
        self.gradcam_target_layer = self.backbone.layer4[-1]

    def forward(self, img_tensor, gender_tensor):
        features = self.backbone(img_tensor)
        gender_tensor = gender_tensor.unsqueeze(1)

        fc_input = torch.cat([features, gender_tensor], dim = 1)
        out = self.regression_head(fc_input)

        return out.squeeze(1)
