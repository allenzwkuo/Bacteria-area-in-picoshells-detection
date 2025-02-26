import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
from model import UNET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO('model_weights/yolo_final_model_weights.pt')
unet_model = UNET(in_channels=3, out_channels=1).to(device)
unet_model.load_state_dict(torch.load("model_weights/unet_weights.pth.tar", map_location=torch.device('cpu'))["state_dict"])
unet_model.eval()

def process_image(image_path):
    white_percentages = []
    image_to_analyze = Image.open(image_path)

    if image_to_analyze.mode != 'RGB':
        image_to_analyze = image_to_analyze.convert('RGB')

    results = yolo_model(image_to_analyze, imgsz=2048, task="segment", device=device)
    boxes = results[0].boxes

    def is_square(cropped_image):
        width, height = cropped_image.size
        aspect_ratio = width / height
        return 0.9 <= aspect_ratio <= 1.1

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for idx, box in enumerate(boxes.xyxy):
        x_min, y_min, x_max, y_max = box.tolist()
        cropped_image = image_to_analyze.crop((x_min, y_min, x_max, y_max))

        if is_square(cropped_image):
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
        # Remove outliers using IQR method
        q25, q75 = np.percentile(white_percentages, 25), np.percentile(white_percentages, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        filtered_percentages = [p for p in white_percentages if lower_bound <= p <= upper_bound]

        if filtered_percentages:
            avg_white_percentage = np.mean(filtered_percentages)
            return round(avg_white_percentage, 2)
        else:
            return 0
    else:
        return 0
