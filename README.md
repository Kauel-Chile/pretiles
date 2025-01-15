# Proyecto Pretiles

Proyecto de medicion de pretiles con ia 

## Estructura de Repositorio

- **generator_dataset**: Permite generar un dataset artificial para entrenar la IA.
- **ia**: Código relacionado con el entrenamiento de la IA.
- **service_subsampling**: Servicio en FastAPI que permite subir una nube de puntos, realizar un submuestreo de la nube de puntos y transformarla al formato requerido por Potree.


To DO
- [] mascara de pixeles validos en la nube de puntos
- [] pixeles invalidos deben ser completados con el valor medio
- [] imagen normalizada (restando el valor medio antes de procesar con la red)
- [] imgen debe ser int8 y no float
- [] crear el ciclo de iteracion cargando el zip en ram
- [] crear arquitectura unet (5 niveles)