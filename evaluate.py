import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from args import get_args
from dataset import KneeDatasetTest
from model import UNetLext
from utils import dice_loss_from_logits, check_device
from torch.utils.data import DataLoader


def evaluate(model, loader, device):
    """Evaluate model on loader, return loss + Dice score."""
    os.makedirs("test_predictions", exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch['image'].to(device)

            logits = model(images)
            prob = torch.sigmoid(logits)  # logits to probability with sigmoid
            pred = (prob > 0.5).float()  # binarize probability

            mask = pred.cpu().squeeze().numpy()

            plt.imsave(f"test_predictions/mask_{i}.png", mask, cmap="gray")



def main():
    args = get_args()
    device = check_device()

    print("Using device ", device)

    # Load dataset
    test_data = os.path.join(args.csv_dir, 'test.csv')
    test_data = pd.read_csv(test_data)
    test_dataset = KneeDatasetTest(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model and assign to device
    model = UNetLext(input_channels=1,
                     output_channels=1,
                     pretrained=False,
                     path_pretrained='',
                     restore_weights=False,
                     path_weights=''
                     ).to(device)

    ckpt_path = os.path.join("checkpoints", f"best_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")


    # Loading saved model
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    evaluate(model, test_loader, device)

    print("Saved prediction samples to test_predictions/")


if __name__ == "__main__":
    main()
