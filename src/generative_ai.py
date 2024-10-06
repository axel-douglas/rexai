# src/generative_ai.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Verificar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Cargar datos
df_materiales = pd.read_csv('../data/materials_data.csv')

# Seleccionar columnas relevantes
features = [
    'Resistencia_Tensión',
    'Resistencia_Compresión',
    'Densidad',
    'Conductividad_Térmica',
    'Coeficiente_Expansión_Térmica',
    'Elasticidad',
    'Dureza',
    'Temperatura_Fusión',
    'Conductividad_Eléctrica',
    'Resistencia_Corrosión',
    'Impacto_Ambiental',
    # 'Cluster'  # Excluir 'Cluster' para generar nuevas combinaciones sin sesgo
]

# Escalar características
scaler = StandardScaler()
X = scaler.fit_transform(df_materiales[features])

# Guardar el scaler para uso posterior
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/scaler.pkl')

# Dataset personalizado
class MaterialesDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Crear dataset y dataloader
dataset = MaterialesDataset(X)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definir el modelo generativo (Autoencoder Variacional)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, latent_dim)  # Media
        self.fc32 = nn.Linear(64, latent_dim)  # LogVar
        # Decoder
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)  # Media y LogVar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = torch.relu(self.fc4(z))
        h5 = torch.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Instanciar el modelo y mover a dispositivo
input_dim = X.shape[1]
model = VAE(input_dim).to(device)

# Definir el optimizador y la función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # Kullback-Leibler Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Entrenamiento del modelo
epochs = 100
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(dataset)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Pérdida Promedio: {avg_loss:.4f}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), '../models/rexai_vae.pth')
print("Modelo generativo entrenado y guardado en 'models/rexai_vae.pth'")

# Función para generar nuevas combinaciones
def generar_nuevas_combinaciones(model, scaler, num_samples=10, objetivo=None):
    model.eval()
    with torch.no_grad():
        # Generar muestras aleatorias en el espacio latente
        z = torch.randn(num_samples, model.fc31.out_features).to(device)
        samples = model.decode(z)
        samples = samples.cpu().numpy()
        # Desescalar las muestras
        samples = scaler.inverse_transform(samples)
        df_samples = pd.DataFrame(samples, columns=features)
        # Filtrar por objetivo si se especifica
        if objetivo:
            for key, value in objetivo.items():
                if 'min' in key:
                    prop = key.replace('_min', '')
                    df_samples = df_samples[df_samples[prop] >= value]
                elif 'max' in key:
                    prop = key.replace('_max', '')
                    df_samples = df_samples[df_samples[prop] <= value]
        return df_samples

# Ejemplo de generación de combinaciones optimizadas para alta resistencia
objetivo = {
    'Resistencia_Tensión_min': 400,
    'Densidad_max': 5.0,
    'Impacto_Ambiental_max': 0.5
}
nuevas_combinaciones = generar_nuevas_combinaciones(model, scaler, num_samples=100, objetivo=objetivo)

# Guardar las nuevas combinaciones
nuevas_combinaciones.to_csv('../data/nuevas_combinaciones.csv', index=False)
print("Nuevas combinaciones generadas y guardadas en 'data/nuevas_combinaciones.csv'")
