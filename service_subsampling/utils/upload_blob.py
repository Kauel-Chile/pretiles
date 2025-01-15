from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os

def upload_to_blob_storage(connection_string, container_name, file_path, blob_filename):
    """
    Sube un archivo a Azure Blob Storage.

    :param connection_string: Cadena de conexión a Azure Blob Storage.
    :param container_name: Nombre del contenedor en Blob Storage.
    :param file_path: Ruta del archivo a subir.
    :param blob_name: Nombre del blob en el almacenamiento.
    """
    # Crear el cliente de servicio de blob
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Crear el cliente de contenedor
    container_client = blob_service_client.get_container_client(container_name)
    
    # Crear el cliente de blob
    blob_client = container_client.get_blob_client(blob_filename)
    
    # Subir el archivo
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)

# Ejemplo de uso
# connection_string = "tu_cadena_de_conexion"
# container_name = "tu_contenedor"
# file_path = "ruta/a/tu/archivo.zip"
# blob_name = "nombre_del_blob.zip"
# upload_to_blob_storage(connection_string, container_name, file_path, blob_name)