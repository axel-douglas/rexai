# src/data_simulation.py

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de Seaborn
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

# Obtener la ruta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definir las rutas para las carpetas 'data' y 'assets'
data_dir = os.path.join(script_dir, '..', 'data')
assets_dir = os.path.join(script_dir, '..', 'assets')

# Crear las carpetas 'data' y 'assets' si no existen
os.makedirs(data_dir, exist_ok=True)
os.makedirs(assets_dir, exist_ok=True)

# Lista ampliada de materiales reciclados con fórmulas químicas y propiedades detalladas
materiales = [
    {'Nombre': 'Aluminio Reciclado', 'Fórmula': 'Al', 'Tipo': 'Metal', 'Grupo': 'Metales'},
    {'Nombre': 'Plástico PET Reciclado', 'Fórmula': 'C10H8O4', 'Tipo': 'Polímero', 'Grupo': 'Polímeros'},
    {'Nombre': 'Fibra de Carbono Reciclada', 'Fórmula': 'C', 'Tipo': 'Fibra', 'Grupo': 'Compuestos'},
    {'Nombre': 'Vidrio Reciclado', 'Fórmula': 'SiO2', 'Tipo': 'Vidrio', 'Grupo': 'Cerámicos'},
    {'Nombre': 'Residuos Orgánicos', 'Fórmula': 'Variable', 'Tipo': 'Orgánico', 'Grupo': 'Orgánicos'},
    {'Nombre': 'Titanio Reciclado', 'Fórmula': 'Ti', 'Tipo': 'Metal', 'Grupo': 'Metales'},
    {'Nombre': 'Basalto Reciclado', 'Fórmula': 'Variable', 'Tipo': 'Mineral', 'Grupo': 'Cerámicos'},
    {'Nombre': 'Polietileno Reciclado', 'Fórmula': 'C2H4', 'Tipo': 'Polímero', 'Grupo': 'Polímeros'},
    {'Nombre': 'PVC Reciclado', 'Fórmula': 'C2H3Cl', 'Tipo': 'Polímero', 'Grupo': 'Polímeros'},
    {'Nombre': 'Poliestireno Reciclado', 'Fórmula': 'C8H8', 'Tipo': 'Polímero', 'Grupo': 'Polímeros'},
    # Añade más materiales según sea necesario
]

# Generación de combinaciones y propiedades detalladas
datos_materiales = []

for material in materiales:
    for _ in range(50):  # Generar 50 variaciones por material
        propiedades = {
            'Nombre': material['Nombre'],
            'Fórmula': material['Fórmula'],
            'Tipo': material['Tipo'],
            'Grupo': material['Grupo'],
            'Resistencia_Tensión': round(random.uniform(50, 500), 2),        # MPa
            'Resistencia_Compresión': round(random.uniform(50, 500), 2),     # MPa
            'Densidad': round(random.uniform(0.5, 8.0), 3),                  # g/cm³
            'Conductividad_Térmica': round(random.uniform(0.1, 400), 2),     # W/m·K
            'Coeficiente_Expansión_Térmica': round(random.uniform(1e-6, 20e-6), 9),  # 1/K
            'Elasticidad': round(random.uniform(10, 200), 2),                # GPa
            'Dureza': round(random.uniform(1, 10), 2),                       # Mohs
            'Temperatura_Fusión': round(random.uniform(300, 2000), 2),       # °C
            'Conductividad_Eléctrica': round(random.uniform(0, 60e6), 2),    # S/m
            'Resistencia_Corrosión': round(random.uniform(0, 1), 2),         # Índice
            'Costo': round(random.uniform(0.5, 50), 2),                      # $/kg
            'Disponibilidad': random.choice(['Alta', 'Media', 'Baja']),
            'Impacto_Ambiental': round(random.uniform(0, 1), 2),             # Índice
        }
        datos_materiales.append(propiedades)

# Convertir a DataFrame
df_materiales = pd.DataFrame(datos_materiales)

# Escalar datos numéricos
numeric_cols = df_materiales.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df_scaled = df_materiales.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_materiales[numeric_cols])

# Reducción de dimensionalidad con UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(df_scaled[numeric_cols])

# Añadir resultados de UMAP al DataFrame
df_materiales['UMAP_1'] = embedding[:, 0]
df_materiales['UMAP_2'] = embedding[:, 1]

# Clustering con HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
cluster_labels = clusterer.fit_predict(embedding)
df_materiales['Cluster'] = cluster_labels

# Guardar el DataFrame en un archivo CSV dentro de la carpeta 'data'
csv_path = os.path.join(data_dir, 'materials_data.csv')
df_materiales.to_csv(csv_path, index=False)

print(f"Conjunto de datos generado, clusters asignados y guardado en '{csv_path}'")

# Visualización de clusters
plt.figure(figsize=(12, 8))
palette = sns.color_palette('bright', len(np.unique(cluster_labels)))
sns.scatterplot(
    x='UMAP_1', y='UMAP_2',
    hue='Cluster',
    data=df_materiales,
    legend='full',
    palette=palette
)
plt.title('Clusters de Materiales (UMAP + HDBSCAN)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Guardar la visualización en la carpeta 'assets'
plot_path = os.path.join(assets_dir, 'clusters.png')
plt.savefig(plot_path)
plt.show()

print(f"Visualización de clusters guardada en '{plot_path}'")
