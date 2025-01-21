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
- [ ] iterar dataset artificial
- [ ] grafica de metricas
- [ ] clases para sanjas
- [ ] post procesamiento para detectar el ancho y alto
- [ ] ancho y alto de pretiles que no estan en normal
- [x] probar con imagenes reales