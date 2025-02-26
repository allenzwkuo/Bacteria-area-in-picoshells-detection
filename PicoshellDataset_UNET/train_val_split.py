import os
import random
import shutil

train_imgs_dir = 'train_imgs'
train_masks_dir = 'train_masks'
val_imgs_dir = 'val_imgs'
val_masks_dir = 'val_masks'

all_images = [f for f in os.listdir(train_imgs_dir) if f.endswith('.jpg')]

num_to_move = int(0.25 * len(all_images))

images_to_move = random.sample(all_images, num_to_move)

for img in images_to_move:

    mask_name = img.replace('.jpg', '_mask.jpg')
    
    if os.path.exists(os.path.join(train_masks_dir, mask_name)):
        shutil.move(os.path.join(train_imgs_dir, img), os.path.join(val_imgs_dir, img))
        
        shutil.move(os.path.join(train_masks_dir, mask_name), os.path.join(val_masks_dir, mask_name))
        
        print(f"Moved {img} and its corresponding mask {mask_name}")
    else:
        print(f"Mask not found for {img}, skipping...")

print("Done moving 25% of images and masks.")
