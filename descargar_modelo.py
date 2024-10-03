import gdown
import os
"""
# URL de la carpeta que quieres descargar
url = 'https://drive.google.com/drive/folders/1WI5bFboRYN07Acoj7I1wqpmXYFXVHgDW'

# Descargar la carpeta completa
gdown.download_folder(url, quiet=False)

print("Descarga completada.")"""



output_dir = 'videos_frutas'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# URL de la carpeta de Google Drive
url = 'https://drive.google.com/drive/folders/1WMS-g-gzp-ku1HGs9AW3EonUSBVWb1fJ'

# Descargar la carpeta completa en el directorio videos_frutas
gdown.download_folder(url, output=output_dir, quiet=False)

print("Descarga completada.")
