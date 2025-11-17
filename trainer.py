import torch

from args import get_args
import torch.nn as nn
import torch.optim as optim
from utils import dice_loss_from_logits, save_checkpoint


def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model

    # binary cross entropy
    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_losses = []
    val_losses = []
    val_dices = []

    best_metric = None
    best_path = None

    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for data_batch in train_loader:
            images = data_batch['image'].to(device=device)
            masks = data_batch['mask'].to(device=device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_bce = bce(outputs, masks)

            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_score = validate_model(model, val_loader, bce, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_dices.append(float(val_score))

        print(f"Epoch : {epoch+1}/{args.epochs}"
              f"Train loss: {train_loss:.4f} |"
              f"Val loss: {val_loss:.4f} |"
              f"Val Dice: {val_score:.4f} |")

        best_metric, best_path = save_checkpoint(
            epoch=epoch,
            model=model,
            val_metric=val_score,
            best_val_metric=best_metric,
            prev_model_path=best_path,
            comparator="gt",
            save_dir="checkpoints"
        )

    return train_losses, val_losses, val_dices


def validate_model(model, val_loader, loss_func, device):
    model.eval()

    val_loss = 0.0
    val_score = 0.0

    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch['image'].to(device=device)
            masks = data_batch['mask'].to(device=device)

            outputs = model(images)
            loss_bce = loss_func(outputs, masks)

            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            val_loss += loss.item()
            val_score += 1 - dice_loss_from_logits(outputs, masks)

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_dice = val_score / len(val_loader)

    return val_epoch_loss, val_epoch_dice







