import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import os
import string


model = YOLO('../model_weights/yolo_model_weights.pt')

input_dir = '../Picoshell_images'
output_dir = "../Picoshell_images_sliced"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def is_square(cropped_image):
    width, height = cropped_image.size
    aspect_ratio = width / height
    return 0.9 <= aspect_ratio <= 1.1 

alphabet = string.ascii_uppercase   
counter = 1  

for image_name in os.listdir(input_dir):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(input_dir, image_name)
        image_to_analyze = Image.open(image_path)

        results = model(image_to_analyze, imgsz=1024, task="segment")

        boxes = results[0].boxes

        for box in boxes.xyxy:  
            x_min, y_min, x_max, y_max = box.tolist()  

            cropped_image = image_to_analyze.crop((x_min, y_min, x_max, y_max))

            if is_square(cropped_image):
                letter_index = (counter - 1) // 99
                letter = alphabet[letter_index]  
                num_suffix = (counter - 1) % 99 + 1  
                
                file_name = f"{letter}{str(num_suffix).zfill(2)}.jpg"
                
                cropped_image.save(os.path.join(output_dir, file_name))
                
                print(f"Saved: {file_name} ({x_min}, {y_min}, {x_max}, {y_max})")

                counter += 1
