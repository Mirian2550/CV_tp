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
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    gdown.download_folder(url, output=output_dir, quiet=False)
    print(f"Download completed and replaced the folder: {output_dir}.")


url_other = 'https://drive.google.com/drive/folders/1aSMeKlKso3x9yxIFUz4ZM5tEnL3C9BZp?usp=sharing'
output_dir_other = '../ejercicio_1'
download_from_drive(url_other, output_dir_other)

url_other = 'https://drive.google.com/drive/folders/15vMYnhA1zBOn2Ua69O01ZdbnpkMYLHFn?usp=sharing'
output_dir_other = 'fotos_frutas'
download_from_drive(url_other, output_dir_other)

url_videos = 'https://drive.google.com/drive/folders/1WMS-g-gzp-ku1HGs9AW3EonUSBVWb1fJ'
output_dir_videos = '../videos_frutas'
download_from_drive(url_videos, output_dir_videos)

url_other = 'https://drive.google.com/drive/u/1/folders/1_8HTzIySFfDBcB_quHgIL7w3AVd6fD0F'
output_dir_other = '../_modelos'
download_from_drive(url_other, output_dir_other)
