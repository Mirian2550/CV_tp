import gdown
import os
"""
# URL de la carpeta que quieres descargar
url = 'https://drive.google.com/drive/folders/1WI5bFboRYN07Acoj7I1wqpmXYFXVHgDW'

# Descargar la carpeta completa
gdown.download_folder(url, quiet=False)

print("Descarga completada.")



output_dir = 'videos_frutas'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# URL de la carpeta de Google Drive
url = 'https://drive.google.com/drive/folders/1WMS-g-gzp-ku1HGs9AW3EonUSBVWb1fJ'

# Descargar la carpeta completa en el directorio videos_frutas
gdown.download_folder(url, output=output_dir, quiet=False)

print("Descarga completada.")
"""
import gdown
import os
import shutil


def download_from_drive(url, output_dir):
    """
    Downloads the folder from Google Drive to the specified output directory.
    If the directory already exists, it will be removed and replaced by the new download.

    Parameters:
    - url (str): The URL of the Google Drive folder.
    - output_dir (str): The directory where the folder will be downloaded.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete the existing directory

    os.makedirs(output_dir)  # Create a fresh directory

    gdown.download_folder(url, output=output_dir, quiet=False)
    print(f"Download completed and replaced the folder: {output_dir}.")


# Example usage
url_videos = 'https://drive.google.com/drive/folders/1WMS-g-gzp-ku1HGs9AW3EonUSBVWb1fJ'
output_dir_videos = 'videos_frutas'
download_from_drive(url_videos, output_dir_videos)

url_other = 'https://drive.google.com/drive/u/1/folders/1_8HTzIySFfDBcB_quHgIL7w3AVd6fD0F'
output_dir_other = '_modelos'
download_from_drive(url_other, output_dir_other)
