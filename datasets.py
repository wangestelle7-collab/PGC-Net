import glob
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        if mode == "train":
            A = 'ct'
            B = 'pet'
        if mode == "val":
            A = 'cttest'
            B = 'pettest'
        self.files_A = sorted(glob.glob(os.path.join(root, A) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, B) + '/*.*'))


    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



class ImageDatasetV2(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        if mode == "train":
            A = 'ct'
            B = 'pet'
            C = 'mask'
        if mode == "val":
            A = 'cttest'
            B = 'pettest'
        self.files_A = sorted(glob.glob(os.path.join(root, A) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, B) + '/*.*'))
        if C:
            self.files_C = sorted(glob.glob(os.path.join(root, C) + '/*.*'))

    def __getitem__(self, index):
        if self.files_C:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
            item_C = self.transform(Image.open(self.files_C[index % len(self.files_C)]).convert('RGB'))
            return {'A': item_A, 'B': item_B, 'C': item_C}
        else:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
            return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

