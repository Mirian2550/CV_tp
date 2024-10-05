# Proyecto de Computación Visual

## Materia:

Computación Visual

## Alumnos:

*   Daniel Ponce
*   Mirian Yañez

## Descripción del Proyecto

Este repositorio contiene las soluciones a varios problemas propuestos en el área de la computación visual, utilizando técnicas avanzadas de procesamiento de imágenes y modelos de detección preentrenados. A continuación se presentan los principales problemas abordados y las soluciones implementadas:

### 1) Desenfoque del Fondo en Imágenes, Videos o Cámaras en Tiempo Real

En este ejercicio utilizamos un modelo preentrenado de YOLOv11 para detectar personas en imágenes o videos y desenfocar el fondo mientras mantenemos a las personas enfocadas. Utilizamos el modelo de segmentación YOLO para generar máscaras de las personas detectadas, y aplicamos un filtro de desenfoque en las áreas que no corresponden a las personas. Esta técnica nos permitió cambiar el fondo de forma dinámica, ya sea reemplazándolo con una imagen o un video, o aplicando un desenfoque selectivo. El enfoque con YOLO fue clave para identificar de manera precisa los objetos en la escena y modificar adecuadamente el fondo.
### 2) Estimación del Peso de Frutas (Peras, Bananas, Manzanas y Naranjas)

Desarrollamos un enfoque personalizado para estimar el peso de frutas comunes (peras, bananas, manzanas y naranjas). Utilizando máscaras de segmentación generadas por el modelo, se calculan las áreas de las frutas en la imagen y se aplican fórmulas basadas en geometría y densidad para estimar el peso de cada una de ellas. Este enfoque está afinado para cada tipo de fruta según su forma y densidad.

### 3) Conteo de Frutas en Imágenes y Videos

Para contar la cantidad de frutas que aparecen en imágenes o videos, utilizamos un modelo de detección con un enfoque de seguimiento de objetos mediante ByteTrack. Este método permite no solo detectar las frutas en cada fotograma, sino también llevar un seguimiento continuo de las frutas que se desplazan en videos, evitando contar objetos repetidamente. Esto es ideal para conteo en entornos donde las frutas están en movimiento, como en cintas transportadoras.

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

*   **fine\_tuning/**: Contiene los archivos creados para ajustar los modelos mediante técnicas de fine-tuning, junto con los datasets correspondientes.
*   **src/**: Incluye el código fuente del proyecto escrito en Python. Aquí se encuentran las clases y funciones utilizadas para cada solución.
### happy code :)