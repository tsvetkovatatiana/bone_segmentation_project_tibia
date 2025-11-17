import torch
import matplotlib.pyplot as plt
import json
import os


def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)  # (B,1,H,W)
    targets = targets.float()
    dims = (1,2,3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_score_from_logits(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    dims = (1,2,3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()

import torch


def check_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    return device


def plot_learning_curve(train_losses, val_losses, val_dices):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Loss)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, val_dices, label="Val Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(model, dataloader, device, num_images=3, threshold=0.5):
    model.eval()

    images_shown = 0
    plt.figure(figsize=(10, num_images * 3))

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)     # [B,1,H,W]
            masks  = batch["mask"].to(device)      # [B,1,H,W]

            outputs = model(images)                # logits
            preds = torch.sigmoid(outputs)         # probabilities
            preds = (preds > threshold).float()    # binary

            B = images.shape[0]

            for i in range(B):
                if images_shown >= num_images:
                    plt.tight_layout()
                    plt.show()
                    return

                img  = images[i,0].cpu().numpy()
                mask = masks[i,0].cpu().numpy()
                pred = preds[i,0].cpu().numpy()

                idx = images_shown * 3

                # Input image
                plt.subplot(num_images, 3, idx+1)
                plt.imshow(img, cmap="gray")
                plt.title("X-ray")
                plt.axis("off")

                # Ground truth
                plt.subplot(num_images, 3, idx+2)
                plt.imshow(mask, cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")

                # Prediction
                plt.subplot(num_images, 3, idx+3)
                plt.imshow(pred, cmap="gray")
                plt.title("Prediction")
                plt.axis("off")

                images_shown += 1

    plt.tight_layout()
    plt.show()


def save_checkpoint(epoch, model, val_metric, best_val_metric=None,
                    prev_model_path=None, comparator="gt", save_dir="session"):
    """
    Save the best model checkpoint for each fold based on validation metric.
    Also saves metrics.json with metadata.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Determine if metric improved
    improved = False
    if best_val_metric is None:
        improved = True
    elif comparator == "gt" and val_metric > best_val_metric:
        improved = True
    elif comparator == "lt" and val_metric < best_val_metric:
        improved = True

    # Save new best
    if improved:
        if prev_model_path and os.path.exists(prev_model_path):
            try:
                os.remove(prev_model_path)
            except OSError:
                pass

        ckpt_path = os.path.join(save_dir, f"best_model.pth")
        metrics_path = os.path.join(save_dir, f"metrics.json")

        torch.save(model.state_dict(), ckpt_path)

        metrics = {
            "best_epoch": epoch + 1,
            "best_val_metric": float(val_metric)
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"\nBest model updated — Epoch {epoch+1} | Metric = {val_metric:.4f}")
        print(f"Saved model: {ckpt_path}")

        return val_metric, ckpt_path

    return best_val_metric, prev_model_path
