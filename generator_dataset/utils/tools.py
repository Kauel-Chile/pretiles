import os
import zipfile
from tqdm import tqdm

def get_total_files_scandir(folder_path):
    total_files = 0
    for entry in os.scandir(folder_path):
        if entry.is_file():
            total_files += 1
        elif entry.is_dir():
            total_files += get_total_files_scandir(entry.path)
    return total_files

def add_to_zip(zipf, folder_path, base_folder, pbar):
    for entry in os.scandir(folder_path):
        full_path = entry.path
        relative_path = os.path.relpath(full_path, base_folder)
        if entry.is_file():
            zipf.write(full_path, relative_path)
            pbar.update(1)
        elif entry.is_dir():
            add_to_zip(zipf, full_path, base_folder, pbar)

def zip_folder(folder_path, zip_path):
    total_files = get_total_files_scandir(folder_path)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total_files, desc="Zipping files", unit="file") as pbar:
            add_to_zip(zipf, folder_path, folder_path, pbar)