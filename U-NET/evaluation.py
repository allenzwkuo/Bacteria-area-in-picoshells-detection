import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import UNET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("unet_weights.pth.tar")["state_dict"])
model.eval()

image_paths = ["image.png", "image(1).png", "image(2).png", "image(3).png", "image(4).png"]
images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])

fig, axs = plt.subplots(5, 2, figsize=(12, 30))

for i, image in enumerate(images):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()

    output_np = output.squeeze().cpu().numpy()

    white_pixels = np.sum(output_np == 1)
    total_pixels = output_np.size
    white_percentage = (white_pixels / total_pixels) * 100
    print(f"Image {i+1} - White: {white_percentage:.2f}%")

    axs[i, 0].imshow(image)
    axs[i, 0].set_title(f"Input Image {i+1}")
    axs[i, 0].axis('off')

    axs[i, 1].imshow(output_np, cmap='gray')
    axs[i, 1].set_title(f"White: {white_percentage:.2f}%")
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()
