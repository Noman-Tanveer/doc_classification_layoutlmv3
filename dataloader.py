import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from transformers import AutoProcessor

class DocData(Dataset):
    def __init__(self, img_dir) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.imgs = glob.glob(os.path.join(img_dir, "**/*.png"))
        self.labels = list(map(lambda x:x.strip(), open("labels.txt").readlines()))
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)

    def squeeze_dims(self, encoding):
        for k, v in encoding.items():
            encoding[k] = torch.squeeze(v, dim=0)
        return encoding

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index) -> dict:
        img_path = self.imgs[index]
        image = Image.open(img_path)
        image = image.convert('RGB')
        label = self.labels.index(img_path.split("/")[-2])
        encoding = self.processor(image, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        encoding = self.squeeze_dims(encoding)
        encoding["labels"] = torch.tensor(label)

        return encoding