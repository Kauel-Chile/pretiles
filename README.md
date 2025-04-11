# Proyecto Pretiles

Proyecto de medicion de pretiles con ia 

## Estructura de Repositorio

- **generator_dataset**: Permite generar un dataset artificial para entrenar la IA.
- **ia**: Código relacionado con el entrenamiento de la IA.
- **service_subsampling**: Servicio en FastAPI que permite subir una nube de puntos, realizar un submuestreo de la nube de puntos y transformarla al formato requerido por Potree.


To DO
- [x] mascara de pixeles validos en la nube de puntos
- [x] pixeles invalidos deben ser completados con el valor medio
- [x] imagen normalizada (restando el valor medio antes de procesar con la red)
- [x] crear el ciclo de iteracion cargando el zip en ram
- [x] crear arquitectura unet (5 niveles)
- [x] eval ciclo
- [x] sumar flip h y v
- [x] metricas f1_score
- [x] graficas de losses
- [x] iterar dataset artificial
- [x] probar con imagenes reales
- [x] clases para zanjas en el generador del dataset
- [x] alinear los ejes de las nubes de puntos antes de generar los dem
- [x] generar dataset masivo
- [ ] generar dataset de validacion
- [ ] metricas con la clase zanja
- [x] post procesamiento para detectar el ancho y alto
- [x] ancho y alto de pretiles que no estan en normal
- [x] Encontrar "las rectas" que generar los pretiles
- [x] Exportar el modelo onnx
- [x] Pipeline de inferencia