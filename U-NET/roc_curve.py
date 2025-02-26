import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from model import UNET
from utils import get_loaders
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

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
    TEST_IMG_DIR = "../PicoshellDataset_UNET/test_imgs/"
    TEST_MASK_DIR = "../PicoshellDataset_UNET/test_masks/"

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
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        test_transform,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    all_targets = []
    all_scores = []

    # Collect all predictions and targets
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            # Forward pass
            outputs = model(data)
            scores = torch.sigmoid(outputs)  # Get the probability scores

            # Flatten predictions and targets
            scores_np = scores.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()

            # Store predictions and targets
            all_scores.extend(scores_np)
            all_targets.extend(targets_np)

    # Convert to numpy arrays for easier manipulation
    all_scores = np.array(all_scores)
    all_targets = np.array(all_targets)

    # Separate black and white pixel scores and targets
    black_indices = np.where(all_targets == 0)[0]
    white_indices = np.where(all_targets == 1)[0]

    # Downsample black pixels to the number of white pixels
    num_white = len(white_indices)
    downsampled_black_indices = np.random.choice(black_indices, num_white, replace=False)

    # Combine downsampled black indices with all white indices
    balanced_indices = np.concatenate((downsampled_black_indices, white_indices))

    # Shuffle the indices to mix black and white pixels
    np.random.shuffle(balanced_indices)

    # Get the balanced scores and targets
    balanced_scores = all_scores[balanced_indices]
    balanced_targets = all_targets[balanced_indices]

    # Compute ROC curve using the downsampled scores
    fpr, tpr, thresholds = roc_curve(balanced_targets, balanced_scores)
    roc_auc = auc(fpr, tpr)

    # Debugging: Check the distribution of the downsampled data
    print("Number of black pixels after downsampling:", len(downsampled_black_indices))
    print("Number of white pixels:", num_white)
    print("Balanced targets distribution:", np.bincount(balanced_targets.astype(int)))

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Downsampled Black Pixels')
    plt.legend(loc="lower right")
    plt.show()

    # Split the scores based on ground truth (0 = black, 1 = white)
    black_scores = [score for score, target in zip(all_scores, all_targets) if target == 0]
    white_scores = [score for score, target in zip(all_scores, all_targets) if target == 1]

    # Plot histograms for black and white pixels
    plt.figure(figsize=(10, 5))

    # Black pixels histogram (ground truth 0)
    plt.subplot(1, 2, 1)
    plt.hist(black_scores, bins=200, color='black', edgecolor='gray')
    plt.xlabel('Predicted Probability (y-scores)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Scores (Black Pixels)')

    # White pixels histogram (ground truth 1)
    plt.subplot(1, 2, 2)
    plt.hist(white_scores, bins=200, color='white', edgecolor='gray')
    plt.xlabel('Predicted Probability (y-scores)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Scores (White Pixels)')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
