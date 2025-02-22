import os
from PIL import Image

# Define input and output directories
input_dir = 'raw_images_unsplit'
output_dir = 'large_split_images'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to split an image into 9 parts
def split_image(image_path, output_dir):
    image = Image.open(image_path)
    width, height = image.size
    if width != height:
        print(f"Skipping {image_path}, as it's not a square image.")
        return

    # Calculate the size of each smaller image
    split_size = width // 3

    # Split the image into 9 smaller images
    count = 1
    for i in range(3):
        for j in range(3):
            left = j * split_size
            upper = i * split_size
            right = left + split_size
            lower = upper + split_size
            cropped_image = image.crop((left, upper, right, lower))
            # Create a name for the cropped image and save it
            base_name = os.path.basename(image_path)
            file_name = f"{os.path.splitext(base_name)[0]}_{count}.png"
            cropped_image.save(os.path.join(output_dir, file_name))
            count += 1
    print(f"Saved 9 split images for {os.path.basename(image_path)}.")

# Iterate through all images in the input directory and split them
for image_name in os.listdir(input_dir):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        image_path = os.path.join(input_dir, image_name)
        split_image(image_path, output_dir)

print("All images have been split and saved.")
