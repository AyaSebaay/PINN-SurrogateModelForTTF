import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from typing import List, Tuple, Dict
import random
import pandas as pd
import time
from datetime import timedelta
import torch.multiprocessing as mp
import json

class CSVSubjectNifti3DDataset(Dataset):
    def __init__(self, subject_ids):
       
        self.subject_ids = subject_ids
        # Collect all file pairs for the specified subjects and file indices
        self.metadata = [] # Store (subject_id, file_index) for each sample
        self.file_pairs = []  
        for subject_id in subject_ids:
            
            for i in range(1, 6):
                cond_file = f"C:/Dataset For training/Conductivity maps/{subject_id}/conductivity_map_{i}.nii.gz"
                volt_file = f"C:/Dataset For training/voltage/{subject_id}/voltage_map_{i}.nii.gz"
                
                
                self.file_pairs.append((cond_file , volt_file))
                self.metadata.append((subject_id, i))
        
        print(f"Found {len(self.file_pairs)} conductivity-voltage pairs for {len(subject_ids)} subjects")
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        conductivity_file, voltage_file = self.file_pairs[idx]
        conductivity_img = nib.load(conductivity_file)
        conductivity_data = conductivity_img.get_fdata().astype(np.float32)

        voltage_img = nib.load(voltage_file)
        voltage_data = voltage_img.get_fdata().astype(np.float32)
        
        conductivity_tensor = torch.from_numpy(conductivity_data)
        voltage_tensor = torch.from_numpy(voltage_data)
        
        # Create boundary mask (where voltage is not zero)
        boundary_mask = (voltage_tensor != 0)
        
        subject_id, file_index = self.metadata[idx]
        
        
        
        return {
            'conductivity_map': conductivity_tensor,
            'voltage_bc': voltage_tensor,
            'boundary_mask': boundary_mask,
            'subject_id': subject_id,
            'file_index': file_index,
        }



class DoubleConv3D(nn.Module):
    """(3D convolution => LeakyReLU) * 2 with same padding"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv with skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(out_channels * 2, out_channels) # we can put in_channel in first input

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatches automatically
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        
        # Apply padding if upsampled tensor is smaller
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        
        # Input and skip connection now have the same size
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class Surrogate3DPINN(nn.Module):
    """3D U-Net with padding to maintain spatial dimensions"""
    def __init__(self):
        super(Surrogate3DPINN, self).__init__()
        
        # Encoder
        self.inc = DoubleConv3D(2, 8)
        self.down1 = Down3D(8, 16)
        self.down2 = Down3D(16, 32)
        self.down3 = Down3D(32, 64)
        self.down4 = Down3D(64, 128)
        
        # Decoder
        self.up1 = Up3D(128, 64)
        self.up2 = Up3D(64, 32)
        self.up3 = Up3D(32, 16)
        self.up4 = Up3D(16, 8)
        
        # Output
        self.outc = nn.Conv3d(8, 1, kernel_size=1)

    def forward(self, conductivity, voltage_bc):

        x = torch.cat([conductivity, voltage_bc], dim=1)  # [batch, 2, D, H, W]
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)

def finite_difference_gradients_3d(potential: torch.Tensor, dx: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 3D gradients using finite differences"""
    # Initialize gradient tensors
    grad_x = torch.zeros_like(potential)
    grad_y = torch.zeros_like(potential)
    grad_z = torch.zeros_like(potential)
    
    # x-gradient (central difference for interior, forward/backward for boundaries)
    grad_x[:, :, 1:-1, :, :] = (potential[:, :, 2:, :, :] - potential[:, :, :-2, :, :]) / (2 * dx)
    grad_x[:, :, 0, :, :] = (potential[:, :, 1, :, :] - potential[:, :, 0, :, :]) / dx
    grad_x[:, :, -1, :, :] = (potential[:, :, -1, :, :] - potential[:, :, -2, :, :]) / dx
    
    # y-gradient
    grad_y[:, :, :, 1:-1, :] = (potential[:, :, :, 2:, :] - potential[:, :, :, :-2, :]) / (2 * dx)
    grad_y[:, :, :, 0, :] = (potential[:, :, :, 1, :] - potential[:, :, :, 0, :]) / dx
    grad_y[:, :, :, -1, :] = (potential[:, :, :, -1, :] - potential[:, :, :, -2, :]) / dx
    
    # z-gradient
    grad_z[:, :, :, :, 1:-1] = (potential[:, :, :, :, 2:] - potential[:, :, :, :, :-2]) / (2 * dx)
    grad_z[:, :, :, :, 0] = (potential[:, :, :, :, 1] - potential[:, :, :, :, 0]) / dx
    grad_z[:, :, :, :, -1] = (potential[:, :, :, :, -1] - potential[:, :, :, :, -2]) / dx
    
    return grad_x, grad_y, grad_z

def physics_loss_3d(potential: torch.Tensor, conductivity_map: torch.Tensor, dx: float = 0.001) -> torch.Tensor:
    """Compute ∇·(σ∇φ) = 0 residual in 3D using finite differences"""
    # Compute gradients ∇φ
    grad_x, grad_y, grad_z = finite_difference_gradients_3d(potential, dx)
    
    # Compute σ∇φ
    sigma_grad_x = conductivity_map * grad_x
    sigma_grad_y = conductivity_map * grad_y
    sigma_grad_z = conductivity_map * grad_z
    
    # Compute divergence ∇·(σ∇φ)
    div_x, _, _ = finite_difference_gradients_3d(sigma_grad_x, dx)
    _, div_y, _ = finite_difference_gradients_3d(sigma_grad_y, dx)
    _, _, div_z = finite_difference_gradients_3d(sigma_grad_z, dx)
    
    divergence = div_x + div_y + div_z
    
    # Physics residual (should be zero)
    physics_residual = torch.mean(divergence**2)
    
    return physics_residual

def boundary_loss_3d(potential: torch.Tensor, voltage_bc: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
    """Loss for 3D boundary conditions"""
    # Only consider points where boundary_mask is True
    boundary_pred = potential[boundary_mask]
    boundary_true = voltage_bc[boundary_mask]
    
    return torch.mean((boundary_pred - boundary_true)**2)

def load_and_split_subjects(csv_file_path , train_ratio = 0.8, random_state = 42):
    df = pd.read_csv(csv_file_path)
    subject_ids = df['Subject ID']
    
    # Set a seed for reproducibility
    np.random.seed(random_state)
    # Shuffle the list and split it
    train_ids, test_ids = np.split(np.random.permutation(subject_ids), [int(train_ratio * len(subject_ids))])
    return train_ids, test_ids

def train_model_3d(model: nn.Module, train_loader: DataLoader, 
                  optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int = 1000):
    """Train the 3D PINN model with validation and loss tracking"""
    model.train()
    
    # Initialize lists to store losses
    train_losses = []
    #val_losses = []
    train_physics_losses = []
    train_boundary_losses = []
    #val_physics_losses = []
    #val_boundary_losses = []
    epochs_list = []
    
    # Create directory for saving loss data
    save_dir = "D:/Results"

    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_physics = 0.0
        epoch_train_boundary = 0.0
        epoch_train_total = 0.0
        
        for batch in train_loader:
            # Move data to device
            voltage_bc = batch['voltage_bc'].to(device).unsqueeze(1)     # [B, 1, D, H, W]
            
            conductivity = batch['conductivity_map'].to(device).unsqueeze(1)  # [B, 1, D, H, W]
            boundary_mask = batch['boundary_mask'].to(device).unsqueeze(1)   # [B, 1, D, H, W]

            
            # Forward pass
            potential_pred = model(conductivity, voltage_bc)
            # Compute losses
            p_loss = physics_loss_3d(potential_pred, conductivity, dx=0.001)
            b_loss = boundary_loss_3d(potential_pred, voltage_bc, boundary_mask)
            
            # Combined loss
            total_loss = 1e-20 * p_loss +  b_loss
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # Accumulate losses
            epoch_train_physics += 1e-20 * p_loss
            epoch_train_boundary += b_loss
            epoch_train_total += total_loss
            #print(1e-20 * epoch_train_physics)
            #print(epoch_train_boundary)
            #print(total_loss)
        
        # Calculate averages
        avg_train_physics = (epoch_train_physics / len(train_loader)).item()
        avg_train_boundary = (epoch_train_boundary / len(train_loader)).item()
        avg_train_total = (epoch_train_total / len(train_loader)).item()
       
        # Store losses for plotting
        train_losses.append(avg_train_total)
        train_physics_losses.append(avg_train_physics)
        train_boundary_losses.append(avg_train_boundary)
       
        epochs_list.append(epoch)
        
        # Print statistics
        print(f'Epoch {epoch:4d}/{num_epochs}')
        print(f'  Train - Total: {avg_train_total:.6f}, Physics: {avg_train_physics:.6f}, Boundary: {avg_train_boundary:.6f}')
       
        #if(avg_train_total < 0.01):
            #break;
        
        # Save checkpoint every 100 epochs
        #if epoch % 100 == 0:
            #checkpoint = {
                #'epoch': epoch,
                #'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                #'train_loss': avg_train_total,
                #'val_loss': avg_val_total,
            #}
            #torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    #calclate training time
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    
    # Save all loss data to files
    loss_data = {
        'epochs': epochs_list,
        'train_losses': train_losses,
        'train_physics_losses': train_physics_losses,
        'train_boundary_losses': train_boundary_losses,
        'training time': total_time_str}
        
    
    
    # Save as JSON
    with open(os.path.join(save_dir, 'loss_data4.json'), 'w') as f:
        json.dump(loss_data, f, indent=4)

    print(f"All loss data saved to directory: {save_dir}")
    
    return model, optimizer, loss_data
    
def save_model(model, optimizer, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Configuration
    csv_file_path = "E:/Datasets/Stage 2/add electrode folder/All Subjects .csv"
    batch_size = 1
    num_epochs = 40
    learning_rate = 1e-4
    train_ratio = 0.8
    
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and split subject IDs
    print("Loading and splitting subject IDs from CSV...")
    train_subjects, test_subjects = load_and_split_subjects(csv_file_path, train_ratio)
    
    # Set random seeds
    #torch.manual_seed(42)
    #torch.manual_seed(12345)
    torch.manual_seed(67890)
    
    # Create datasets
    train_dataset = CSVSubjectNifti3DDataset(train_subjects)
    test_dataset = CSVSubjectNifti3DDataset(test_subjects)
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True , num_workers = 2, prefetch_factor = 2 , pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False )
    
    
    model = Surrogate3DPINN().to(device)
    model = nn.DataParallel(model)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    load_model(model , optimizer, "D:/Results/train3.pth")
    # Train the model
    print("Starting 3D training with U-Net architecture...")
    model, optimizer, train_losses = train_model_3d(model, train_loader, optimizer, device, num_epochs)
    save_model(model, optimizer,"D:/Results/train4.pth" )
    print("3D U-Net model saved")
    # Save the model
    #model_name = "unet_3d.pth"
    #torch.save(model.state_dict(), model_name)
    #print(f"3D U-Net model saved as '{model_name}'")



