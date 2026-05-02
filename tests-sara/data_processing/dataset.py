import torch
from PIL import Image
from os import path, listdir, exists

""" BAA Dataset Class

- Takes as input: df (Pandas DataFrame), tranforms (Torchvision transforms)

- DataFrame required columns:
    - "path": path to image
    - "male": patient's sex/gender (1.0 if male, 0.0 otherwise)
    - "boneage": patient's real age (in months)
"""
class BAA_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms = None):
        self.df = df
        self.n = df.shape[0]
        self.transforms = transforms

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        
        if not exists(row["path"]):
            return None, None, None
        
        img = Image.open(row["path"])

        if self.transforms is not None:
            img = self.transforms(img)
        
        gender = torch.tensor(row["male"], dtype=torch.float32)
        age = torch.tensor(row["boneage"], dtype=torch.float32)

        return img, gender, age
    
    def __len__(self):
        return self.n
    
    def set_transforms(self, transforms):
        self.transforms = transforms
    
    def get_random_items(self, n_samples=1):
        output = []
        rows = self.df.sample(n_samples)

        for _, row in rows.iterrows():
            img = Image.open(row["path"])
            if self.transforms is not None:
                img = self.transforms(img)
            gender = torch.tensor(row["male"], dtype=torch.float32)
            age = torch.tensor(row["boneage"], dtype=torch.float32)
            idx = torch.tensor(row["id"], dtype=torch.int32)
            output.append( (img, gender, age, idx) )

        return output
