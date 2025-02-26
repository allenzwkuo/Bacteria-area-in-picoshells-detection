from PIL import Image
import os

input_dir = '../picoshell_images_sliced_edgecases' 

for image_name in os.listdir(input_dir):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(input_dir, image_name)

        img = Image.open(image_path)

        img_resized = img.resize((100, 100))

        img_resized.save(image_path)

        print(f"Resized and saved: {image_name}")
