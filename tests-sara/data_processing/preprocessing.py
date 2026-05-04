from torchvision import transforms
from torchvision.transforms.functional import adjust_contrast, adjust_brightness





def contrast():
    def _func(img):
        return adjust_contrast(img, contrast_factor= 10)
    return _func

def get_transforms(target_size, color_channels_nb = 1):
    norm_mean = (0.5)
    norm_std = (0.5)
    if color_channels_nb == 3:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=color_channels_nb),
        transforms.ToTensor(),
        transforms.Resize(target_size), 
        transforms.Normalize(mean=norm_mean, std=norm_std),
        #contrast(),
    ])