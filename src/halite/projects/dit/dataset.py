import functools
import pickle
import io

import bagz
import numpy as np
from PIL import Image
from torchvision import datasets


class ImageFolder(datasets.ImageFolder):
    @functools.wraps(datasets.ImageFolder.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.samples = sorted(self.samples)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


class CenterCrop:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, img):
        return center_crop_arr(img, self.image_size)


class BagzDataset:
    def __init__(self, data_path: str, transform=None):
        self._reader = bagz.Reader(data_path)
        self.transform = transform

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, index):
        record = pickle.loads(self._reader[index])
        image = Image.open(io.BytesIO(record["image"])).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, record["class_id"]

