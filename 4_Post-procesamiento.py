import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import torch
import torch.nn as nn
import joblib
import plotly.express as px
import os

# Configuración de la página
st.set_page_config(
    page_title="Rex-AI: Innovando Materiales para el Espacio",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ocultar menú y pie de página de Streamlit
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Logo y Título
st.markdown(
    """
    <div class="header">
        <img src="https://i.imgur.com/your-logo.png" alt="Rex-AI Logo" width="150">
        <h1 class="title">Rex-AI</h1>
        <h2 class="subtitle">Innovando Materiales para el Espacio</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Descripción del Proyecto Rex-AI
st.markdown(
    """
    <div class="description">
        <p><strong>Rex-AI</strong> utiliza inteligencia artificial para transformar residuos en materiales suprareciclados, optimizados para aplicaciones espaciales y sustentabilidad ambiental. Aquí visualizamos el impacto de los datos satelitales para detectar basurales y optimizar la producción de nuevos materiales.</p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# Cargar archivos geoespaciales
@st.cache_data
def cargar_shapefiles():
    gdf_basurales = gpd.read_file('./data/shp/gt/gt_basurales_4326.geojson')
    gdf_aoi_predict = gpd.read_file('./data/shp/aoi-predict.geojson')
    return gdf_basurales, gdf_aoi_predict

gdf_basurales, gdf_aoi_predict = cargar_shapefiles()

# Visualización interactiva de basurales detectados
st.markdown("<h2 class='section-title'>Visualización de Basurales Detectados</h2>", unsafe_allow_html=True)
st.markdown("<div class='description'>Aquí visualizamos los basurales detectados mediante imágenes satelitales y procesadas por el modelo de IA de Rex-AI.</div>", unsafe_allow_html=True)

# Mapa de Basurales con Folium
mapa_basurales = folium.Map(location=[-34.6037, -58.3816], zoom_start=10)

# Añadir las zonas de basurales al mapa
folium.GeoJson(gdf_basurales, name="Basurales Detectados").add_to(mapa_basurales)

# Añadir zonas AOI para predicción
folium.GeoJson(gdf_aoi_predict, style_function=lambda x: {'color': 'green'}, name="Áreas de Predicción (AOI)").add_to(mapa_basurales)

# Mostrar máscaras de predicción generadas
@st.cache_data
def cargar_predicciones():
    predicciones = glob.glob('./data/predictions/*.tif')
    return predicciones

predicciones = cargar_predicciones()

# Agregar una capa para cada predicción
for pred_file in predicciones:
    layer_name = os.path.basename(pred_file).replace('_pred.tif', '')
    folium.raster_layers.ImageOverlay(
        image=pred_file,
        name=f"Predicción {layer_name}",
        opacity=0.6
    ).add_to(mapa_basurales)

# Añadir control de capas
folium.LayerControl().add_to(mapa_basurales)

# Mostrar el mapa en el dashboard
st_data = st_folium(mapa_basurales, width=700, height=500)

# Sección de materiales suprareciclados
st.markdown("<h2 class='section-title'>Optimización de Materiales Basados en Datos Satelitales</h2>", unsafe_allow_html=True)
st.markdown("<div class='description'>Usamos los datos satelitales para ajustar las combinaciones de materiales suprareciclados, enfocándonos en las regiones con mayor acumulación de residuos.</div>", unsafe_allow_html=True)

@st.cache_data
def cargar_datos_materiales():
    df = pd.read_csv('./data/materials_data.csv')
    return df

df_materiales = cargar_datos_materiales()

# Visualización interactiva con Plotly
fig_clusters = px.scatter(
    df_materiales, x='UMAP_1', y='UMAP_2',
    color='Cluster',
    hover_data=['Nombre', 'Tipo'],
    template='plotly_dark',
    width=1000,
    height=600
)
st.plotly_chart(fig_clusters, use_container_width=True)

# Generación de Nuevas Combinaciones
st.markdown("<h2 class='section-title'>Generación de Nuevas Combinaciones de Materiales</h2>", unsafe_allow_html=True)

# Cargar modelo generativo y scaler
modelo_path = './models/rexai_vae.pth'
scaler_path = './models/scaler.pkl'

if os.path.exists(modelo_path) and os.path.exists(scaler_path):
    class VAE(nn.Module):
        def __init__(self, input_dim, latent_dim=10):
            super(VAE, self).__init__()
            # Encoder
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc31 = nn.Linear(64, latent_dim)
            self.fc32 = nn.Linear(64, latent_dim)
            # Decoder
            self.fc4 = nn.Linear(latent_dim, 64)
            self.fc5 = nn.Linear(64, 128)
            self.fc6 = nn.Linear(128, input_dim)

        def encode(self, x):
            h1 = torch.relu(self.fc1(x))
            h2 = torch.relu(self.fc2(h1))
            return self.fc31(h2), self.fc32(h2)

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

    scaler = joblib.load(scaler_path)
    input_dim = len(scaler.mean_)
    model = VAE(input_dim)
    model.load_state_dict(torch.load(modelo_path, map_location=torch.device('cpu')))
    model.eval()

    # Definir los parámetros de entrada
    st.markdown("<div class='description'><h3>Defina los parámetros para generar nuevas combinaciones de materiales</h3></div>", unsafe_allow_html=True)
    with st.form(key='params_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            resistencia_min = st.number_input("Resistencia a la Tensión Mínima (MPa):", min_value=0.0, max_value=1000.0, value=300.0)
        with col2:
            densidad_max = st.number_input("Densidad Máxima (g/cm³):", min_value=0.0, max_value=10.0, value=5.0)
        with col3:
            impacto_max = st.number_input("Impacto Ambiental Máximo (Índice 0-1):", min_value=0.0, max_value=1.0, value=0.5)
        num_samples = st.slider("Número de muestras a generar:", min_value=1, max_value=100, value=10)
        submit_button = st.form_submit_button(label='Generar Nuevas Combinaciones')

    if submit_button:
        def generar_nuevas_combinaciones(model, scaler, num_samples=10, objetivo=None):
            model.eval()
            with torch.no_grad():
                z = torch.randn(num_samples, model.fc31.out_features)
                samples = model.decode(z)
                samples = samples.numpy()
                samples = scaler.inverse_transform(samples)
                df_samples = pd.DataFrame(samples, columns=scaler.feature_names_in_)
                if objetivo:
                    for key, value in objetivo.items():
                        if 'min' in key:
                            prop = key.replace('_min', '')
                            df_samples = df_samples[df_samples[prop] >= value]
                        elif 'max' in key:
                            prop = key.replace('_max', '')
                            df_samples = df_samples[df_samples[prop] <= value]
                return df_samples

        objetivo = {
            'Resistencia_Tensión_min': resistencia_min,
            'Densidad_max': densidad_max,
            'Impacto_Ambiental_max': impacto_max
        }

        nuevas_combinaciones = generar_nuevas_combinaciones(model, scaler, num_samples=int(num_samples), objetivo=objetivo)

        if not nuevas_combinaciones.empty:
            st.markdown("<div class='description'><h3>Nuevas Combinaciones Generadas</h3></div>", unsafe_allow_html=True)
            st.dataframe(nuevas_combinaciones)
            csv = nuevas_combinaciones.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Nuevas Combinaciones",
                data=csv,
                file_name='nuevas_combinaciones.csv',
                mime='text/csv',
                key='download-csv'
            )
        else:
            st.warning("No se encontraron combinaciones que cumplan con los objetivos especificados.")
else:
    st.warning("El modelo no está entrenado. Ejecuta 'generative_ai.py' primero.")

# Footer
st.markdown(
    """
    <hr>
    <div class='footer'>
    © 2024 Rex-AI. Todos los derechos reservados.
    </div>
    """,
    unsafe_allow_html=True
)
