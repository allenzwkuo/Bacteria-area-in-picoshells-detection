import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# defines dataset class for image loading
class PicoshellDatatset(Dataset):

    # initialize dataset parameters
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir # directory for images
        self.mask_dir = mask_dir # directory for masks
        self.transform = transform # tranformations to apply
        self.images = os.listdir(image_dir) # list of all images 

    # compute num samples in dataset
    def __len__(self):
        return len(self.images)

    # retrieve image and corresponding mask by index based on name
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # all corresponding masks named by appending "_mask.jpg" to the end of image names
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg","_mask.jpg")) 

        # loads image in RGB format
        image = np.array(Image.open(img_path).convert("RGB"))
        # convert image to numpy array for model processing 
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # binarizes mask, converts all white areas to 1
        mask[mask == 255.0] = 1.0

        # applies transformations if specified on both masks and images
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
