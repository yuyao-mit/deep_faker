import torch
from torch import nn


class BaseModel(nn.Module):
    """
    Base class for all models. Provides common utility methods and serves as a template for derived models.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        """
        Forward pass to be overridden by subclasses.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_weights(self):
        """
        Initializes weights for all layers using Kaiming initialization.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class CNN_DF(BaseModel):
    def __init__(self):
        super(CNN_DF, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=4, padding=1),  
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=8, stride=4, padding=2), 
            nn.Sigmoid(), 
        )

    def decode(self, z):
        z = z + torch.randn_like(z) * 0.1  
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decode(encoded)
        return {"encoded": encoded, "decoded": decoded}


class AE_DF(BaseModel):
    def __init__(self, input_dim=128 * 128, latent_dim=16):
        """
        Simple autoencoder with fully connected layers.

        Args:
            input_dim (int): Flattened input size (default is for 64x64 grayscale images).
            latent_dim (int): Dimensionality of the latent space.
        """
        super(AE_DF, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64,latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8192),
            nn.ReLU(),
            nn.Linear(8192, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1) 

        encoded = self.encoder(x)
        encoded = encoded + torch.randn_like(encoded) * 0.1  
        decoded = self.decoder(encoded)
        decoded = decoded.view(batch_size, 1, 128, 128) 

        return {"encoded": encoded, "decoded": decoded}
        
    '''
    def forward(self, x):
    """
    Forward pass through the AE_DF model.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, 1, 128, 128].
    Returns:
        dict: Encoded latent representation and reconstructed image.
    """
    print(f"Input device: {x.device}, Model device: {next(self.parameters()).device}")
    batch_size = x.size(0)

    # Ensure input is on the correct device
    x = x.to(next(self.parameters()).device)

    # Flatten input
    x = x.view(batch_size, -1)  # Flatten to [batch_size, input_dim]

    # Encoding
    encoded = self.encoder(x)
    encoded = encoded + torch.randn_like(encoded).to(encoded.device) * 0.1  # Add noise to the latent space

    # Decoding
    decoded = self.decoder(encoded)
    decoded = decoded.view(batch_size, 1, 128, 128)  # Reshape to image dimensions

    return {"encoded": encoded, "decoded": decoded}
    '''

