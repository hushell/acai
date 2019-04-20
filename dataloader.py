from __future__ import print_function, division
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class QMUL_dataset(Dataset):
    """Queen Mary Sketch Dataset."""

    def __init__(self, root_dir, train=True, image_size=32):
        """
        Args:
        root_dir (str): Directory with all images.
        image_size (int): Network input image size.
        transforms (callabel, optional): Optional transformation applied to samples.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(.5, .5, .5))
        self.photo_transforms = transforms.Compose([transforms.Resize(size=(image_size, image_size)), transforms.ToTensor(), normalize])
        self.sketch_transforms = transforms.Compose([transforms.Resize(size=(image_size, image_size)), transforms.ToTensor(), normalize])
        if train:
            photo_file = 'photo_train.txt'
            sketch_file = 'sketch_train.txt'
        else:
            photo_file = 'photo_test.txt'
            sketch_file = 'sketch_test.txt'
        with open(os.path.join(root_dir, photo_file), 'r') as f:
            photos = [line.rstrip('\n') for line in f]
        self.photos = [os.path.join('ShoeV2_photo', w) for w in photos]
        with open(os.path.join(root_dir, sketch_file), 'r') as f:
            sketches = [line.rstrip('\n') for line in f]
        sketches = [w.replace('svg', 'png') for w in sketches]
        self.sketches = [os.path.join('ShoeV2_sketch', w) for w in sketches]

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        # read sketch randomly
        sketch_idx = np.random.randint(0, len(self.sketches))
        sketch = Image.open(os.path.join(self.root_dir, self.sketches[sketch_idx]))
        sketch = self.sketch_transforms(sketch)

        # read photo randomly
        photo_idx = np.random.randint(0, len(self.photos))
        photo = Image.open(os.path.join(self.root_dir, self.photos[photo_idx]))
        photo = self.photo_transforms(photo)

        return photo, sketch

## Data loader
def TrainValDataLoader(root_dir, train=True, image_size=32, batchSize=64):

	dataSet = QMUL_dataset(root_dir, train, image_size)
	dataLoader = DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=train, num_workers=1, drop_last = True)

	return dataLoader
if __name__ == "__main__":
    d = QMUL_dataset(root_dir = '/space_sde/ShoeV2_F', train=False)
    print(len(d))

    import sys
    test_loader = TrainValDataLoader('/space_sde/ShoeV2_F', train=False)
    for i, data in enumerate(test_loader):
        photo, sketch = data
        print(photo.size())
        print(sketch.size())
        if i == 0:
            sys.exit()
