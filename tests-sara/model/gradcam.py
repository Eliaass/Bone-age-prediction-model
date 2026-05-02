import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

""" Grad-CAM algorithm
- get 2 things from last conv layer:
    -  activations, during forward pass
    - gradients, during backward pass
- let it run both passes
- compute heatmap
"""

class GradCAM:

    def __init__(self, model, target_layer):
        
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.register_hooks()

    def register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image, gender):

        self.model.eval()

        output = self.model(image, gender)

        self.model.zero_grad()
        output.backward()

        grads = self.gradients
        activations = self.activations

        weights = grads.mean(dim = (2,3), keepdim = True)
        cam = (weights * activations).sum(dim = 1, keepdim = True)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def resize_cam(self, cam, image_size):

        # cam.shape = (1,1,x,y)
        #cam_resized = cv2.resize(cam, (image.shape[3], image.shape[2]))

        # print(f"cam: dtype {cam.dtype}, shape: {cam.shape}")

        cam_resized = F.interpolate(
            cam,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )

        
        cam_resized = cam_resized.squeeze().detach().numpy()
        #print(f"cam resized: dtype {cam_resized.dtype}, shape: {cam_resized.shape}")
        
        return cam_resized

        
