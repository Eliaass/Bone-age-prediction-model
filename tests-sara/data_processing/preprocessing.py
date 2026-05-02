from torchvision import transforms
from torchvision.transforms.functional import adjust_contrast, adjust_brightness





def contrast():
    def _func(img):
        return adjust_contrast(img, contrast_factor= 10)
    return _func

def get_transforms(target_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize(target_size), 
        transforms.Normalize(mean=(0.5), std=(0.5)),
        #contrast(),
    ])