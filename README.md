# Rex-AI: Waste Mapping and Supra-Recycling Platform

## Purpose

**Rex-AI** is a geospatial and machine learning-based platform designed to identify and map open-air landfills and optimize the development of supra-recycled materials. This application integrates satellite data processing, machine learning, and material science to not only locate waste sites but also turn waste into high-value materials for advanced applications, including aerospace.

Rex-AI aligns with global sustainability goals by efficiently managing waste and optimizing recycling processes, using cutting-edge AI to revolutionize the recycling industry at a global scale.

## Functionalities

### 1. **Open-Air Landfill Detection**

Rex-AI processes satellite imagery (such as Sentinel-2 data) to automatically detect and monitor open-air landfills using advanced machine learning models. This module provides real-time insights on waste accumulation in different regions, enabling proactive waste management strategies.

#### Key Technologies:
- **Satellite Data (Sentinel-2)**: High-resolution, multiband satellite imagery used for identifying waste sites.
- **Machine Learning (UNet)**: A convolutional neural network (CNN) optimized for image segmentation, enabling precise identification of landfill areas.
- **Geospatial Analysis (GDAL, Orfeo Toolbox)**: Tools for preprocessing and analyzing satellite data.

### 2. **Generation of Supra-Recycled Materials**

Rex-AI leverages a **Variational Autoencoder (VAE)** to develop new combinations of materials from recycled waste. By optimizing material properties such as tensile strength, density, and environmental impact, Rex-AI helps convert waste into high-performance materials suitable for aerospace and other advanced industries.

#### Key Features:
- **VAE Model**: Generates new material combinations optimized for specific applications.
- **Interactive Interface (Streamlit)**: Users can set material parameters and visualize the generated results.
- **Cluster Analysis (UMAP, HDBSCAN)**: Identifies patterns in material properties, helping users understand relationships between different materials.

### 3. **Geospatial Visualization of Waste Flows**

Rex-AI provides real-time, interactive maps of waste movement across different regions. By integrating satellite data and machine learning predictions, the platform allows users to visualize waste generation, transport, and accumulation.

#### Visual Tools:
- **Interactive Maps (Folium)**: Real-time tracking of waste accumulation sites and flow dynamics.
- **Dashboards (Streamlit + Plotly)**: Dynamic visualizations of landfill data and supra-recycled material properties.

## Key Modules

### **1. Preprocessing of Satellite Data**
- **Goal**: Convert raw satellite images into usable datasets.
- **Process**: Generate images in multiple bands (RGB, NIR, SWIR, NDSW) and calculate the Normalized Difference ShortWave (NDSW) index to detect organic waste accumulation.
- **Tools**: GDAL, Orfeo Toolbox, `satproc_extract_chips`.

### **2. Machine Learning Model Training**
- **Goal**: Train UNet models to detect open-air landfills from satellite images.
- **Process**: Use segmented image chips and corresponding masks to teach the model to identify landfill areas.
- **Tools**: UNet architecture, supervised learning with labeled datasets.

### **3. Material Generation and Clustering**
- **Goal**: Generate new supra-recycled materials optimized for specific industries using a Variational Autoencoder.
- **Process**: Users can define material parameters, and Rex-AI generates new materials, which are then visualized through interactive clustering and performance metrics.
- **Tools**: VAE, UMAP for dimensionality reduction, HDBSCAN for clustering.

## Impact

### **Environmental Impact**
- **Waste Reduction**: By detecting and analyzing open-air landfills, Rex-AI helps communities and industries manage waste more effectively, reducing environmental contamination and improving public health.
- **Sustainable Materials**: The platform's ability to generate high-quality materials from waste promotes circular economy practices and reduces dependency on virgin materials.

### **Industrial Application**
- **Advanced Materials**: Rex-AI enables the development of supra-recycled materials optimized for high-performance industries, including aerospace, defense, and construction.
- **Data-Driven Waste Management**: Governments and companies can leverage Rex-AI's geospatial data and machine learning models to design more efficient waste management systems.

