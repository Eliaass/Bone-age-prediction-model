from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch

class simple_BAA_model(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained resnet18
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        last_width = self.cnn.fc.in_features

        # make it take grayscale instead of RGB: need to replace first conv layer
        c = self.cnn.conv1
        self.cnn.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = c.out_channels,
            kernel_size = c.kernel_size, 
            stride = c.stride, 
            padding = c.padding, 
            bias = c.bias
        )

        # replace classifier with regression head: replace cnn.fc with useless layer and store fc elsewhere (for adding gender input to reg head later)
        self.cnn.fc = nn.Identity()
        self.regression_head = nn.Linear(last_width + 1, 1)

    def forward(self, img_tensor, gender_tensor):
        features = self.cnn(img_tensor)
        gender_tensor = gender_tensor.unsqueeze(1)

        fc_input = torch.cat([features, gender_tensor], dim = 1)
        out = self.regression_head(fc_input)

        return out.squeeze(1)
