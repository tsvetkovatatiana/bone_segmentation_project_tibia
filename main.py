from torch.utils.data import DataLoader
from args import get_args
import os
import pandas as pd
from dataset import KneeDataset
from model import UNetLext
from trainer import train_model
from utils import check_device, plot_learning_curve, plot_predictions


def main():
    args = get_args()

    device = check_device()
    print(f"Using {device}")

    # Step 1: reading csv
    train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
    val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

    # Step 2: preparing our dataset
    print("Preparing dataset")
    train_dataset = KneeDataset(train_set)
    val_dataset = KneeDataset(val_set)

    # Step 3: initialising the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialising the model
    model = UNetLext(input_channels=1,
                     output_channels=1,
                     pretrained=False,
                     path_pretrained='',
                     restore_weights=False,
                     path_weights=''
    ).to(device)

    train_losses, val_losses, val_dices = train_model(model, train_loader, val_loader, device)

    plot_learning_curve(train_losses, val_losses, val_dices)

    plot_predictions(model, val_loader, device, num_images=5)


if __name__ == '__main__':
    main()