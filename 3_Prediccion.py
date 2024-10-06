import torch
import torch.nn as nn
import torch.nn.functional as F
from osgeo import gdal
import numpy as np
import tifffile as tiff
import os
import glob
from tqdm import tqdm
from PIL import Image  # Asegúrate de que esta línea esté incluida

# Definir un tamaño fijo más pequeño para las imágenes (128x128 para reducir la memoria)
IMAGE_SIZE = (128, 128)

# Arquitectura U-Net
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

# Función para redimensionar imágenes
def resize_image(image_array, size=IMAGE_SIZE):
    """Redimensionar imagen a un tamaño fijo."""
    return np.array(Image.fromarray(image_array).resize(size, resample=Image.BILINEAR))

# Función para cargar imágenes y redimensionarlas
def load_image(image_path):
    img = tiff.imread(image_path)  # Cargar imagen TIFF
    img = resize_image(img)  # Redimensionar
    img = np.expand_dims(img, axis=0)  # Añadir una dimensión de canal
    img = torch.from_numpy(img).float().unsqueeze(0)  # Convertir a tensor y añadir dimensión batch
    return img.to(device)

# Función para guardar las predicciones
def save_prediction(pred_mask, output_path):
    pred_mask = np.squeeze(pred_mask)  # Eliminar dimensiones innecesarias
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Convertir a binario
    tiff.imwrite(output_path, pred_mask)  # Guardar como archivo .tif

# Cargar el modelo entrenado
device = torch.device("cpu")  # Usar CPU para evitar problemas de memoria
model = UNet().to(device)
model.load_state_dict(torch.load('./models/unet_trained.pth', map_location=device))  # Cargar el modelo entrenado
model.eval()

# Directorio de imágenes de prueba y de salida
test_images = glob.glob('./data/processed/train/*.tif')  # Usar las imágenes de entrenamiento
pred_output_dir = './data/predictions/'
os.makedirs(pred_output_dir, exist_ok=True)

# Realizar predicciones para todas las imágenes de prueba
for img_file in tqdm(test_images):
    # Cargar la imagen y predecir la máscara
    image = load_image(img_file)
    with torch.no_grad():
        pred_mask = model(image)

    # Guardar la predicción
    pred_mask = pred_mask.cpu().numpy()
    pred_file = os.path.join(pred_output_dir, os.path.basename(img_file).replace('.tif', '_pred.tif'))
    save_prediction(pred_mask, pred_file)

print("Predicciones completadas y guardadas en:", pred_output_dir)
