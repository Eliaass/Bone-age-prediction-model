from torchvision import transforms


def get_transforms(target_size = None, include_resize = False, color_channels_nb = 1):
    norm_mean = (0.5)
    norm_std = (0.5)
    if color_channels_nb == 3:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    if include_resize:

        return transforms.Compose([
            #transforms.Grayscale(num_output_channels=color_channels_nb),
            transforms.ToTensor(),
            transforms.Resize(target_size), 
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        return transforms.Compose([
            #transforms.Grayscale(num_output_channels=color_channels_nb),
            #transforms.ToTensor(), 
            #transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

