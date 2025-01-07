import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# hyper parameters

LEARNING_RATE = 1e-4 # step size (generic 1e-3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # hardware
NUM_EPOCHS = 1 
NUM_WORKERS = 2 # hardware
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
PIN_MEMORY = True # hardware
LOAD_MODEL = True
TRAIN_IMG_DIR = "../PicoshellDataset_UNET/train_imgs/"
TRAIN_MASK_DIR = "../PicoshellDataset_UNET/train_masks/"
VAL_IMG_DIR = "../PicoshellDataset_UNET/val_imgs/"
VAL_MASK_DIR = "../PicoshellDataset_UNET/val_masks/"

# training function
def train_fn(loader, model, optimizer, loss_fn, scaler):

    # initialize tqdm
    loop = tqdm(loader)

    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()

        epoch_loss += loss.item()

        # update tqdm 
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)

# validation function with backprop and gradient descent
def validate_fn(loader, model, loss_fn):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            targets = targets.float().unsqueeze(1).to(DEVICE)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            epoch_loss += loss.item()

    model.train()
    return epoch_loss / len(loader)

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
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

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform, val_transform,
        NUM_WORKERS, PIN_MEMORY,
    )

    if LOAD_MODEL:
        checkpoint = torch.load("unet_weights.pth.tar", map_location=DEVICE)
        # load_checkpoint(torch.load("unet_weights.pth.tar"), model)
        load_checkpoint(checkpoint, model)
        
    check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.amp.GradScaler('cuda')

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = validate_fn(val_loader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}")
        print(f"Epoch {epoch} complete")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

    # loss plot
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and validation loss")
    plt.show()

if __name__ == "__main__":
    main()