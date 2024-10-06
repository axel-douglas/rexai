import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import glob
import os
from osgeo import gdal
import numpy as np
import torch.nn.functional as F

# Definir un tamaño fijo para las imágenes (por ejemplo, 256x256)
IMAGE_SIZE = (256, 256)

# Arquitectura del modelo U-Net
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output layer
        output = torch.sigmoid(self.output_layer(dec1))
        return output

# Dataset personalizado con redimensionamiento
class SatelliteDataset(Dataset):
    def __init__(self, images, masks, image_size=IMAGE_SIZE):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        image = gdal.Open(self.images[idx])
        mask = gdal.Open(self.masks[idx])
        
        # Redimensionar imágenes y máscaras al tamaño fijo
        image = self.resize_image(image)
        mask = self.resize_image(mask)

        image = self.transform(image)
        mask = self.transform(mask)
        return image, mask

    def resize_image(self, gdal_dataset):
        """Redimensiona una imagen GDAL a un tamaño fijo."""
        warp_options = gdal.WarpOptions(format='MEM', width=self.image_size[1], height=self.image_size[0])
        image_resized = gdal.Warp('', gdal_dataset, options=warp_options)
        img_array = image_resized.ReadAsArray()
        return img_array

    def __len__(self):
        return len(self.images)

# Cargar datos
train_images = glob.glob('./data/processed/train/*.tif')
train_masks = glob.glob('./data/masks/*.tif')

# Asegurarse de que hay datos suficientes
if len(train_images) == 0 or len(train_masks) == 0:
    raise ValueError("No se encontraron imágenes o máscaras de entrenamiento.")

# Crear el dataset y dataloader
train_dataset = SatelliteDataset(train_images, train_masks)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Definir el modelo U-Net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# Definir el optimizador y la función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Loop de entrenamiento
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

# Guardar el modelo
os.makedirs('./models', exist_ok=True)
torch.save(model.state_dict(), './models/unet_trained.pth')
