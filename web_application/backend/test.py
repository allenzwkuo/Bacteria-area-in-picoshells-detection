import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
from model import UNET

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
yolo_model = YOLO('model_weights/yolo_final_model_weights.pt')

# Load U-Net model
unet_model = UNET(in_channels=3, out_channels=1).to(device)
unet_model.load_state_dict(torch.load("model_weights/unet_weights.pth.tar")["state_dict"])
unet_model.eval()

# Input directory
input_dir = 'test_images'

# Square check function
def is_square(cropped_image):
    width, height = cropped_image.size
    aspect_ratio = width / height
    return 0.9 <= aspect_ratio <= 1.1 

# U-Net Transform
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])

def convert_to_jpg(image_path):
    if not image_path.lower().endswith('.jpg'):
        img = Image.open(image_path)
        jpg_path = os.path.splitext(image_path)[0] + '.jpg'
        img.convert('RGB').save(jpg_path, 'JPEG')
        return jpg_path
    return image_path

# Processing Images
for image_name in os.listdir(input_dir):
    if image_name.lower().endswith(('.png', '.jpeg', '.tif', '.tiff', '.jpg')):
        image_path = os.path.join(input_dir, image_name)
        jpg_path = convert_to_jpg(image_path)

        image_to_analyze = Image.open(jpg_path)

        # Run YOLO
        results = yolo_model(image_to_analyze, imgsz=2048, task="segment", device="cuda")
        
        boxes = results[0].boxes
        white_percentages = []

        # Loop through each box and process separately
        for idx, box in enumerate(boxes.xyxy):
            x_min, y_min, x_max, y_max = box.tolist()
            cropped_image = image_to_analyze.crop((x_min, y_min, x_max, y_max))

            # Only process square-like boxes
            if is_square(cropped_image):
                # Process with U-Net
                image_tensor = transform(cropped_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = unet_model(image_tensor)
                    output = torch.sigmoid(output)
                    output = (output > 0.5).float()

                output_np = output.squeeze().cpu().numpy()
                white_pixels = np.sum(output_np == 1)
                total_pixels = output_np.size
                white_percentage = (white_pixels / total_pixels) * 100
                
                if white_percentage >= 1:
                    white_percentages.append(white_percentage)

        if white_percentages:
            avg_white_percentage = np.mean(white_percentages)
            print(f"Average white percentage (above 1%) for image {image_name}: {avg_white_percentage:.2f}%")
        else:
            print(f"No boxes had white percentage above 1% for image {image_name}")

        if jpg_path != image_path:
            os.remove(jpg_path)
