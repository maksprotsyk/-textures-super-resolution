import os
import cv2
from torch.utils.data import Dataset
from albumentations import HorizontalFlip, RandomCrop, CenterCrop, ShiftScaleRotate
from albumentations.core.composition import Compose
import random
from src.utils.utils import pad_image
import torchvision.transforms.functional as F

from torchvision.transforms import transforms
import torch

class SuperResDataset(Dataset):
    """
    Dataset for super-resolution training
    """

    def __init__(self, img_dir, mode: str = "train", test_set_size: int = 10):
        """
        :param img_dir: path to dir with images
        :param mode: 'train' or 'test'
        """
        self.mode = mode
        self.img_dir = img_dir
        self.image_shape = (128, 128)

        print(f"{self.image_shape=}")
        print(f"path {self.img_dir}")
        # transformations
        self.transform = self.get_transform()

        # train/test split
        self.test_set_size = test_set_size
        images_names = os.listdir(img_dir)
        random.seed(1)
        random.shuffle(images_names)
        if self.mode == 'train':
            self.data = images_names[self.test_set_size:]
        elif self.mode == 'test':
            self.data = images_names[:self.test_set_size]
        else:
            print(f"Invalid mode {self.mode}")

        # normalization
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data[idx])

        # if self.mode == 'test':
        print(img_path)

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) / 255.0

        # padding image if it's smaller than desired shape
        if (image.shape[0] < self.image_shape[0]) or (image.shape[1] < self.image_shape[1]):
            image = pad_image(image, desired_shape=self.image_shape)

        # TODO: normalize
        image = self.transform(image=image)['image']
        image = self.norm(image)

        downscaled = F.resize(image, size=int(self.image_shape[0] / 2))


        return downscaled.type(torch.FloatTensor), image.type(torch.FloatTensor)

    def get_transform(self):
        if self.mode == 'train':
            return Compose([
                HorizontalFlip(),
                ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                RandomCrop(height=self.image_shape[0], width=self.image_shape[1])
            ])
        else:
            return Compose([
                HorizontalFlip(),
                ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                CenterCrop(height=self.image_shape[0], width=self.image_shape[1])
            ])


if __name__ == "__main__":
    d = SuperResDataset("/home/bohdan/Documents/UCU/3/AI/textures_super-resolution/data/minecraft_textures_v1",
                        test_set_size=50)
    print(len(d))
