from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torchvision
from PIL import Image
import torch
import os
import random
import pandas as pd
from typing import Any, Callable, List, Optional, Union, Tuple


class CelebAExt(torchvision.datasets.CelebA):
    def __init__(
            self,
            root_dir: str,
            selected_attrs: Union[List[str]],
            split: str = "train",
            transform: Optional[Callable] = None
    ) -> None:
        """Initialize and preprocess the CelebA dataset."""
        super(CelebAExt, self).__init__(root=root_dir, split=split,
                target_type="attr", transform=transform, download=False)

        indices = []
        for idx, x in enumerate(self.attr_names):
            if (x in selected_attrs):
                indices.append(idx)

        self.attr = self.attr[:, indices]

        self.num_images = len(self.filename)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename = self.filename[index]
        label = self.attr[index]
        image = Image.open(os.path.join(self.root, self.base_folder,
        "img_align_celeba", filename))
        return self.transform(image), torch.FloatTensor(label.float())
 
    def __len__(self):
        """Return the number of images."""
        return self.num_images
 

def get_loader(root_dir, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, split='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if split == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebAExt(root_dir, selected_attrs, split, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(split=='train'),
                                  num_workers=num_workers)
    return data_loader
