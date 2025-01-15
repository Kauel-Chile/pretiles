import os
import zipfile

def zip_folder(folder_path, output_path):
    """
    Comprime una carpeta en un archivo ZIP.

    :param folder_path: Ruta de la carpeta a comprimir.
    :param output_path: Ruta del archivo ZIP de salida.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

# Ejemplo de uso
# zip_folder('/ruta/a/tu/carpeta', '/ruta/a/tu/archivo.zip')