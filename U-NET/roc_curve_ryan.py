import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from model import UNET
from utils import get_loaders
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load("unet_weights.pth.tar")["state_dict"])
    model.eval()

    BATCH_SIZE = 16
    NUM_WORKERS = 2
    IMAGE_HEIGHT = 100
    IMAGE_WIDTH = 100
    PIN_MEMORY = True
    TEST_IMG_DIR = "../PicoshellDataset_UNET/test_imgs/"  # Path to test images
    TEST_MASK_DIR = "../PicoshellDataset_UNET/test_masks/"  # Path to test masks

    # Define the same transform as used during validation
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Load the test set
    _, test_loader = get_loaders(
        TEST_IMG_DIR,  # Test images directory
        TEST_MASK_DIR,  # Test masks directory
        TEST_IMG_DIR,  # Test images directory (duplicate, but should not cause issues)
        TEST_MASK_DIR,  # Test masks directory (duplicate, but should not cause issues)
        BATCH_SIZE,
        test_transform,  # Use the same transform as during validation
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    all_targets = []
    all_predictions = []
    all_scores = []  # List to store predicted scores

    # With thresholding applied
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            # Forward pass
            outputs = model(data)
            scores = torch.sigmoid(outputs)  # Get the probability scores
            predictions = (scores > 0.5).float()  # Apply threshold to predictions

            # Flatten predictions, targets, and scores
            scores_np = scores.cpu().numpy().flatten()
            predictions_np = predictions.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()

            # Store predictions, targets, and scores
            all_predictions.extend(predictions_np)
            all_targets.extend(targets_np)
            all_scores.extend(scores_np)

    # Split the scores based on ground truth (0 = black, 1 = white)
    black_scores = [score for score, target in zip(all_scores, all_targets) if target == 0]
    black_targets = [target for target in all_targets if target == 0]
    
    white_scores = [score for score, target in zip(all_scores, all_targets) if target == 1]
    white_targets = [target for target in all_targets if target == 1]

    # Normalize the contributions of black and white pixels
    black_weight = len(black_scores) / (len(black_scores) + len(white_scores))  # Weight for black pixels
    white_weight = len(white_scores) / (len(black_scores) + len(white_scores))  # Weight for white pixels

    # Compute ROC curve for black pixels
    fpr_black, tpr_black, _ = roc_curve(black_targets, black_scores)
    roc_auc_black = auc(fpr_black, tpr_black)

    # Compute ROC curve for white pixels
    fpr_white, tpr_white, _ = roc_curve(white_targets, white_scores)
    roc_auc_white = auc(fpr_white, tpr_white)

    # Normalize AUC based on class size
    weighted_roc_auc = black_weight * roc_auc_black + white_weight * roc_auc_white

    # Plot ROC curve as scatter plot
    plt.figure()

    # Plot black pixels ROC curve
    plt.scatter(fpr_black, tpr_black, color='blue', s=30, label=f'Black Pixels ROC curve (AUC = {roc_auc_black:.2f})')

    # Plot white pixels ROC curve
    plt.scatter(fpr_white, tpr_white, color='red', s=30, label=f'White Pixels ROC curve (AUC = {roc_auc_white:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Black and White Pixels (Weighted AUC = {weighted_roc_auc:.2f})')
    plt.legend(loc="lower right")
    plt.show()

    # Plot histograms for black and white pixels
    plt.figure(figsize=(10, 5))

    # Black pixels histogram (ground truth 0)
    plt.subplot(1, 2, 1)
    plt.hist(black_scores, bins=200, color='blue', edgecolor='gray')
    plt.xlabel('Predicted Probability (y-scores)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Scores (Black Pixels)')

    # White pixels histogram (ground truth 1)
    plt.subplot(1, 2, 2)
    plt.hist(white_scores, bins=200, color='red', edgecolor='gray')
    plt.xlabel('Predicted Probability (y-scores)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Scores (White Pixels)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
