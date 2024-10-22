import json
import numpy as np
import os
import cv2
import os.path
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=lambda x:x, dct_status=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.dct_status = dct_status

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])

        if self.dct_status:
            img = cv2.imread(str(image_path))
            img = np.array(img, dtype='uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
        else:
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])
