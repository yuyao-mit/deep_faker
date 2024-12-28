import os
import random
import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter


def set_seed(seed=42):
    """
    Set seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker with a seed.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)

class MultiscaleImageDataset(Dataset):
    """
    Custom dataset for loading multiscale images stored in .tiff format.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory containing the raw images (./Multiscale_Image_Dataset/raw/).
            transform (callable, optional): Optional transform to be applied on each image.
        """
        self.root_dir = root_dir
        self.transform = transform        
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".tiff")
        ]

        if not self.image_paths:
            raise FileNotFoundError(f"No .tiff images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image




class AugmentedMultiscaleImageDataset(Dataset):
    """
    Augmented dataset for loading and augmenting multiscale images.
    """
    def __init__(self, root_dir, transform=None, target_size=1000):
        """
        Args:
            root_dir (str): Directory containing the raw images (./Multiscale_Image_Dataset/raw/).
            transform (callable, optional): Transform to apply to each image.
            target_size (int): Total number of augmented images to create. Default is 10000.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size        
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".tiff")
        ]

        if not self.image_paths:
            raise FileNotFoundError(f"No .tiff images found in {root_dir}")

        self.base_size = len(self.image_paths) 

    def __len__(self):
        return self.target_size

    def __getitem__(self, idx):        
        torch.manual_seed(idx)
        random.seed(idx)
        
        original_idx = idx % self.base_size
        img_path = self.image_paths[original_idx]
        image = Image.open(img_path).convert("L")
        
        augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((128, 128), pad_if_needed=True),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Lambda(lambda img: img if random.random() > 0.5 else img.filter(ImageFilter.GaussianBlur(radius=1))),
        ])

        augmented_image = augmentation_transform(image)
        
        if self.transform:
            augmented_image = self.transform(augmented_image)

        return augmented_image


def create_data_loader(root_dir, batch_size=32, target_size=1000, mode="train"):
    """
    Creates a DataLoader for training or evaluation.

    Args:
        root_dir (str): Directory containing the raw dataset.
        batch_size (int): Batch size for the DataLoader. Default is 32.
        target_size (int): Total number of augmented images to create. Default is 10000.
        mode (str): Mode for the DataLoader ("train" or "eval"). Default is "train".

    Returns:
        DataLoader: A DataLoader instance for the dataset.
    """
    if mode not in ["train", "eval"]:
        raise ValueError(f"Mode must be 'train' or 'eval', got {mode}.")

    # Define transformations
    
    bright_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda img: transforms.functional.adjust_brightness(img, 3)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])


    bright_dataset = MultiscaleImageDataset(root_dir=dataset_dir, transform=bright_transform)
    bright_loader = DataLoader(bright_dataset,
                               batch_size=batch_size,
                               shuffle=(mode=="train"),
                               num_workers=4,
                               worker_init_fn=worker_init_fn)
    return bright_loader
