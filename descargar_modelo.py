import gdown

# URL de la carpeta que quieres descargar
url = 'https://drive.google.com/drive/folders/1WI5bFboRYN07Acoj7I1wqpmXYFXVHgDW'

# Descargar la carpeta completa
gdown.download_folder(url, quiet=False)

print("Descarga completada.")
