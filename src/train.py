import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

def train_deepfaker(model, train_loader, checkpoint_dir, num_epochs=50000, criterion=None, batch_size=32, device=None, checkpoint_interval=100):
    """
    Trains the DeepFaker model and saves checkpoints at specified intervals with multi-GPU support.

    Args:
        model (torch.nn.Module): The DeepFaker model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        checkpoint_dir (str): Directory to save checkpoints.
        num_epochs (int): Number of epochs to train. Default is 100.
        criterion (callable): Loss function. Default is MSELoss().
        batch_size (int): Batch size for training. Default is 32.
        device (torch.device): Device for training (e.g., 'cpu' or 'cuda'). Default is 'cuda' if available.
        checkpoint_interval (int): Save a checkpoint every `checkpoint_interval` epochs. Default is 10.

    Returns:
        float: Total training time in seconds.
    """
    
    if criterion is None:
        criterion = nn.MSELoss()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    epoch_losses = []

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for images in train_loader:
            images = images.to(device)

            outputs = model(images)["decoded"]
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": epoch_losses,
                "seed": 42,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    return total_training_time
