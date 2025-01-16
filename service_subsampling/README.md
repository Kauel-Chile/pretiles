# Service Subsampling cloud points

## Crear un ACI

### Crear imagen del contenedor y subirla a ACR

1. Inicia sesión en Docker con las credenciales del ACR:
   ```sh
   docker login cloudpointscontainerregistry.azurecr.io 
   ```
2. Construye la imagen del contenedor:
   ```sh
   docker build -t service_subsampling:1.x .
   ```
3. Etiqueta la imagen para el ACR:
   ```sh
   docker tag service_subsampling:1.x cloudpointscontainerregistry.azurecr.io/service_subsampling:1.x
   ```
4. Sube la imagen al ACR:
   ```sh
   docker push cloudpointscontainerregistry.azurecr.io/service_subsampling:1.x
   ```

### Crear instancia del contenedor

1. Ejecuta el siguiente comando para crear una instancia del contenedor:

```sh
  az container create \
     --resource-group cloudpoints-frontend_group \
     --name subsampling \
     --image cloudpointscontainerregistry.azurecr.io/service_subsampling \
     --ports 8000 \
     --cpu 1 \
     --memory 1 \
     --os-type Linux \
     --ip-address public \
     --dns-name-label cloudpointservice
```

