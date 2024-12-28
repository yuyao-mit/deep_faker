import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def eval_deepfaker(model, dataset, n_images=5, image_size=128):
    """
    Evaluates the DeepFaker model by reconstructing a few images and visualizing them in a row.

    Args:
        model (torch.nn.Module): The trained DeepFaker model.
        dataset (torch.utils.data.Dataset): The dataset for evaluation.
        n_images (int): Number of images to reconstruct and compare.
        image_size (int): Size of each image (assumes square images, default: 128).
    """
    # Get a batch of original images
    original_imgs = torch.stack([dataset[i] for i in range(n_images)])  # Stack n_images for evaluation
    with torch.no_grad():
        original_imgs = original_imgs.to(next(model.parameters()).device)
        res = model(original_imgs)
        reconstructed_imgs = res['decoded'].cpu() 

    # Plot original and reconstructed images in a row
    fig, axes = plt.subplots(2, n_images, figsize=(15, 5))  # 2 rows: original, reconstructed

    for i in range(n_images):
        # Original image
        original_image = original_imgs[i].squeeze().cpu().numpy()  # Squeeze and convert to NumPy
        axes[0, i].imshow(original_image, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')

        # Reconstructed image
        reconstructed_image = reconstructed_imgs[i].squeeze().numpy()  # Squeeze and convert to NumPy
        axes[1, i].imshow(reconstructed_image, cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_deepfaker(checkpoint_dir, test_loader, model):
    """
    Evaluates the DeepFaker model using the latest checkpoint and visualizes results.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        test_loader (DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): The DeepFaker model to evaluate.
    """
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found!")

    # List and identify the latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in the specified directory!")

    # Find the latest checkpoint based on the epoch number in the filename
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split("_")[-1].split(".")[0]))
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading latest checkpoint: {latest_checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint_path)

    # Load the model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")

    # Plot training loss over epochs
    losses = checkpoint["losses"]
    epochs = list(range(1, len(losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label="Training Loss")
    min_loss_idx = losses.index(min(losses))
    plt.scatter(epochs[min_loss_idx], min(losses), color="red",
                label=f"Lowest Loss: {min(losses):.4f} (Epoch {epochs[min_loss_idx]})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate and visualize reconstructed and latent images
    eval_deepfaker(model, test_loader.dataset, n_images=20, image_size=128)
