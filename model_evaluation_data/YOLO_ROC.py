import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch

model = YOLO('../model_weights/yolo_model_weights.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

input_dir = 'large_split_images'
current_dir = os.path.dirname(os.path.realpath(__file__))

for conf in [round(i * 0.1, 1) for i in range(11)]:
    output_dir = os.path.join(current_dir, f"conf_{conf}")
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        if image_name.endswith('.png'):
            image_path = os.path.join(input_dir, image_name)
            image_to_analyze = Image.open(image_path)
            results = model(image_to_analyze, imgsz=683, task="segment", conf=conf, device=device)
            num_boxes = len(results[0].boxes.xyxy)
            draw = ImageDraw.Draw(image_to_analyze)
            for box in results[0].boxes.xyxy:
                x_min, y_min, x_max, y_max = box.tolist()
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            file_name = f"{os.path.splitext(image_name)[0]}_{num_boxes}.png"
            image_to_analyze.save(os.path.join(output_dir, file_name))
            print(f"Saved: {file_name} with {num_boxes} bounding boxes in folder 'conf_{conf}'")

print("All images processed and saved in respective confidence folders with bounding box count in filenames.")
