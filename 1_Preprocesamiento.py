import os
import glob
from osgeo import gdal, ogr
from tqdm import tqdm

# Rutas de los datos
shapefiles_path = './data/shp/zonas/'  # Directorio con shapefiles de zonas
gt_geojson = './data/shp/gt/gt_basurales_4326.geojson'  # Archivo GeoJSON de los basurales
output_image_path = './data/processed/train/'  # Ruta donde guardaremos las imágenes rasterizadas
output_mask_path = './data/masks/'  # Ruta para las máscaras

# Crear las carpetas de salida si no existen
os.makedirs(output_image_path, exist_ok=True)
os.makedirs(output_mask_path, exist_ok=True)

# Rasterizar shapefiles y generar máscaras
def rasterize_shapefiles(shapefiles_path, output_image_path, output_mask_path, gt_geojson):
    shapefiles = glob.glob(os.path.join(shapefiles_path, '*.shp'))
    
    for shp in shapefiles:
        base_name = os.path.splitext(os.path.basename(shp))[0]
        output_raster = os.path.join(output_image_path, f'{base_name}.tif')
        output_mask = os.path.join(output_mask_path, f'{base_name}_mask.tif')

        # Obtener información del shapefile
        ds = ogr.Open(shp)
        layer = ds.GetLayer()

        # Crear una imagen rasterizada a partir del shapefile
        x_min, x_max, y_min, y_max = layer.GetExtent()
        pixel_size = 0.0001  # Ajusta según sea necesario
        x_res = int((x_max - x_min) / pixel_size)
        y_res = int((y_max - y_min) / pixel_size)

        # Crear el archivo rasterizado
        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
        band = target_ds.GetRasterBand(1)
        band.Fill(0)  # Inicializar el raster con 0 (vacío)
        band.SetNoDataValue(0)

        # Rasterizar el shapefile en el raster creado (imagen principal)
        gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[255])
        target_ds = None

        # Generar máscara (binaria) desde el archivo GeoJSON de basurales
        mask_ds = gdal.GetDriverByName('GTiff').Create(output_mask, x_res, y_res, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
        mask_band = mask_ds.GetRasterBand(1)
        mask_band.Fill(0)  # Inicializar con 0 (vacío)
        mask_band.SetNoDataValue(0)

        # Abrir el archivo GeoJSON y rasterizarlo en la máscara
        geojson_ds = ogr.Open(gt_geojson)
        geojson_layer = geojson_ds.GetLayer()
        gdal.RasterizeLayer(mask_ds, [1], geojson_layer, burn_values=[1])
        mask_ds = None

    print(f"Imágenes generadas en {output_image_path}. Máscaras generadas en {output_mask_path}.")

# Ejecutar la rasterización y generación de máscaras
rasterize_shapefiles(shapefiles_path, output_image_path, output_mask_path, gt_geojson)

print("Preprocesamiento completo.")
