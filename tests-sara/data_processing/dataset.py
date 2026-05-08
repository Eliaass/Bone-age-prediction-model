import torch
from PIL import Image
from os import path, listdir
import numpy as np
from torchvision import transforms

""" BAA Dataset Class

- Takes as input: df (Pandas DataFrame), tranforms (Torchvision transforms)

- DataFrame required columns:
    - "img_path": path to images
    - "mask_path": path to segmentation masks, if they exist
    - "male": patient's sex/gender (1.0 if male, 0.0 otherwise)
    - "boneage": patient's real age (in months)
"""
class BAA_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms = None, apply_segmentation = False):
        self.df = df
        self.n = df.shape[0]
        self.transforms = transforms

        self.apply_segmentation = apply_segmentation
        self.mask_thresh = 120


    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        
        if not path.exists(row["img_path"]):
            return None, None, None
        

        img = self.get_image(row)
        
        gender = torch.tensor(row["male"], dtype=torch.float32)
        age = torch.tensor(row["boneage"], dtype=torch.float32)
        idx = torch.tensor(row["id"], dtype=torch.int32).item()

        return img, gender, age, idx
    
    def __len__(self):
        return self.n

    def get_image(self, row):  

        img = Image.open(row["img_path"]).convert("RGB")


        if self.apply_segmentation:

            mask = Image.open(row["mask_path"]).convert("RGB")

            img = np.array(img)
            mask = np.array(mask)

            img = img * (mask > self.mask_thresh) #apply mask

            img = Image.fromarray(img)


        if self.transforms is not None:
            img = self.transforms(img) #apply transforms


            

        
        return img


    def set_transforms(self, transforms):
        self.transforms = transforms

    
    def get_random_items(self, n_samples=1):
        output = []
        rows = self.df.sample(n_samples)

        for _, row in rows.iterrows():
            img = self.get_image(row)
            gender = torch.tensor(row["male"], dtype=torch.float32)
            age = torch.tensor(row["boneage"], dtype=torch.float32)
            idx = torch.tensor(row["id"], dtype=torch.int32)
            output.append( (img, gender, age, idx.item()) )

        return output
