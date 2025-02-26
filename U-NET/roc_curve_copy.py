import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os
from model import UNET
from utils import get_loaders
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load("unet_weights.pth.tar")["state_dict"])
    model.eval()

    # Define directories
    TEST_IMG_DIR = "../PicoshellDataset_UNET/test_imgs/"  # Path to test images
    TEST_MASK_DIR = "../PicoshellDataset_UNET/test_masks/"  # Path to test masks

    # Define the same transform as used during validation
    test_transform = A.Compose(
        [
            A.Resize(height=100, width=100),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    # Get the list of test image files
    test_img_files = os.listdir(TEST_IMG_DIR)
    test_mask_files = os.listdir(TEST_MASK_DIR)

    thresholds = np.arange(0.0, 1.1, 0.1)
    
    # Initialize lists to store FPR and TPR at each threshold
    fpr_list = []
    tpr_list = []

    # Loop through the images and compute metrics at each threshold
    for img_file, mask_file in zip(test_img_files, test_mask_files):
        # Load the image and mask
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        mask_path = os.path.join(TEST_MASK_DIR, mask_file)

        # Open and transform the image
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')  # Load as binary mask

        # Apply the same transform to both image and mask
        transformed = test_transform(image=np.array(image), mask=np.array(mask))
        data = transformed['image']
        target = transformed['mask']

        data = data.unsqueeze(0).to(device)  # Add batch dimension
        target = target.unsqueeze(0).to(device)  # Add batch dimension

        # Get model predictions (logits)
        with torch.no_grad():
            output = model(data)
            output = torch.sigmoid(output)  # Convert logits to probabilities

        # Flatten predictions and ground truth masks for easier comparison
        output_np = output.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()

        # Loop through each threshold to calculate TPR and FPR
        for threshold in thresholds:
            # Convert probabilities to binary predictions based on threshold
            predictions = (output_np > threshold).astype(int)
            targets = target_np.astype(int)

            # Compute True Positives, False Positives, True Negatives, False Negatives
            TP = np.sum((predictions == 1) & (targets == 1))  # Correctly classified positives
            FP = np.sum((predictions == 1) & (targets == 0))  # False positives
            TN = np.sum((predictions == 0) & (targets == 0))  # True negatives
            FN = np.sum((predictions == 0) & (targets == 1))  # False negatives

            # Calculate the total number of foreground pixels in the ground truth
            total_fg = np.sum(targets)  # Number of white pixels (foreground)
            total_pixels = len(targets)  # Total number of pixels

            # Weigh the metrics based on the number of foreground pixels
            weight_fg = total_fg / total_pixels

            # Calculate TPR (True Positive Rate) and FPR (False Positive Rate) with weighting
            TPR = (TP / (TP + FN)) * weight_fg if (TP + FN) > 0 else 0
            FPR = (FP / (FP + TN)) * (1 - weight_fg) if (FP + TN) > 0 else 0

            # Append to lists for averaging
            fpr_list.append(FPR)
            tpr_list.append(TPR)

    # Convert the lists to numpy arrays
    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)

    # Calculate the average FPR and TPR at each threshold
    avg_fpr = []
    avg_tpr = []

    for i in range(len(thresholds)):
        avg_fpr.append(np.mean(fpr_list[i::len(thresholds)]))  # Average over all images at each threshold
        avg_tpr.append(np.mean(tpr_list[i::len(thresholds)]))

    avg_fpr = np.array(avg_fpr)
    avg_tpr = np.array(avg_tpr)

    # Calculate AUC
    roc_auc = auc(avg_fpr, avg_tpr)
    print(f"AUC: {roc_auc:.2f}")

    # Plot the ROC curve
    plt.figure()
    plt.plot(avg_fpr, avg_tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
