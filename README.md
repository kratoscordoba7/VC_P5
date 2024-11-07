<h1 align="center">🌟 Práctica 5 - Visión por Computador (Curso 2024/2025)</h1>

<img align="left" width="190" height="190" src="./images/octoCat.png"></a>
Se han completado todas las tareas solicitadas de la **Práctica 5** para la asignatura **Visión por Computador**. Detección y caracterización de caras.

*Trabajo realizado por*:

[![GitHub](https://img.shields.io/badge/GitHub-Heliot%20J.%20Segura%20Gonzalez-darkcyan?style=flat-square&logo=github)](https://github.com/kratoscordoba7)

[![GitHub](https://img.shields.io/badge/GitHub-Alejandro%20David%20Arzola%20Saavedra%20-black?style=flat-square&logo=github)](https://github.com/AlejandroDavidArzolaSaavedra)

## 🛠️ Librerías Utilizadas

[![OpenCV](https://img.shields.io/badge/OpenCV-%23FD8C00?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-%2300B6AB?style=for-the-badge&logo=mediapipe)](https://mediapipe.dev/)
[![NumPy](https://img.shields.io/badge/NumPy-%2300A9E0?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Math](https://img.shields.io/badge/Math-%23F0B800?style=for-the-badge&logo=mathworks&logoColor=white)](https://www.mathworks.com/)
[![Random](https://img.shields.io/badge/Random-%23FF5733?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/random.html)
[![Time](https://img.shields.io/badge/Time-%23FF6347?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/time.html)
[![BaseOptions](https://img.shields.io/badge/BaseOptions-%23007A99?style=for-the-badge&logo=python&logoColor=white)](https://mediapipe.dev/)


---
## 🚀 Cómo empezar

Para comenzar con el proyecto, sigue estos pasos:

> [!NOTE]  
> Debes de situarte en un environment configurado como se definió en el cuaderno de la práctica 1  de [otsedom](https://github.com/otsedom/otsedom.github.io/blob/main/VC/P1/README.md#111-comandos-basicos-de-anaconda) y la práctica 5 de [otsedom](https://github.com/otsedom/otsedom.github.io/blob/main/VC/P5/README.md).

### Paso 1: Abrir VSCode y situarse en el directorio:
   
   `C:\Users\TuNombreDeUsuario\anaconda3\envs\VCP5
   
### Paso 2: Clonar y trabajar en el proyecto localmente (VS Code)
1. **Clona el repositorio**: Ejecuta el siguiente comando en tu terminal para clonar el repositorio:
   ```bash
   git clone https://github.com/kratoscordoba7/VCP5.git
   ```
2. Una vez clonado, todos los archivos han de estar situado en el environment del paso 1

### Paso 3: Abrir Anaconda prompt y activar el envioroment:
   ```bash
   conda activate NombreDeTuEnvironment
   ```
Tras estos pasos debería poder ejecutar el proyecto localmente

---

<h2 align="center">📋 Tareas</h2>

### Tarea 1 Detectores faciales y filtros

Tras mostrar opciones para la detección y extracción de información de caras humanas con deepface, la tarea a entregar consiste en proponer un escenario de aplicación y desarrollar un prototipo de temática libre que provoque reacciones a partir de la información extraida del rostro. Los detectores proporcionan información del rostro, y de sus elementos faciales. Ideas inmediatas pueden ser filtros, aunque no hay limitaciones en este sentido. La entrega debe venir acompañada de un gif animado o vídeo de un máximo de 30 segundos con momentos seleccionados de la propuesta.


### Modo Duende

Cuando abrimos la boca, se genera una caída de dinero, la cual representamos mediante una clase que gestiona las coordenadas x e y y asigna un tiempo de expiración para determinar si el objeto sigue descendiendo o desaparece. A continuación, se muestra un fragmento de código que ilustra cómo se implementa este comportamiento:

``` python
# Clase para el dinero
class FallingEmoji:
    def __init__(self, x, y, speed, time_to_live):
        self.x = x
        self.y = y
        self.speed = speed
        self.time_to_live = time_to_live  # Tiempo de vida del emoji
        self.creation_time = time.time()  # Momento en que se creó el emoji

    def update(self):
        self.y += self.speed  # Movimiento hacia abajo
        if self.y > 480:  # Si el emoji se sale de la pantalla, lo reubicamos en la parte superior
            self.y = 0
            self.x = random.randint(0, 640)

        # Verificar si el emoji ha excedido su tiempo de vida
        if time.time() - self.creation_time > self.time_to_live:
            return False  
        return True 
```

Para detectar cuándo debe aparecer el dinero, basta con establecer un umbral de distancia entre los dos puntos centrales de la boca y asignar una probabilidad de que ocurra. De esta manera, podemos determinar el momento en que debe activarse la caída del dinero. A continuación, se muestra un fragmento de código que ilustra cómo se implementa esta lógica:

```python
 # Umbral para la distancia 
 threshold = 40 

 probabilidad_generar_emoji = 0.1

 if mouth_open_distance > threshold and random.random() < probabilidad_generar_emoji:
     if random.random() < 0.4:
         new_emoji = FallingEmoji(random.randint(0, frame.shape[1] - 80), 0, random.randint(2, 5), time_to_live=5)
         falling_emoji.append(new_emoji)
                 
 # Eliminamos segun va pasando el tiempo de vida
 falling_emoji[:] = [emoji for emoji in falling_emoji if emoji.update()]
```

### Modo duende adicional(segmentacion)

En este caso, el funcionamiento es similar al modo duende anterior, pero se le añade segmentación para que el fondo desaparezca y sea reemplazado por una imagen. Esto amplía el uso de MediaPipe, explorando sus diversas funcionalidades. Para la segmentación, utilizamos un modelo que MediaPipe proporciona en su documentación, el cual nos ayudará a segmentar lo que se reproduce a través de la cámara en vivo.

```python
# Configurar las opciones del segmentador
options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path="models/selfie_segmenter.tflite"), 
    output_category_mask=True,
    running_mode=vision.RunningMode.LIVE_STREAM, 
    result_callback=segmentation_callback
)

# Creamos el segmentador
segmenter = vision.ImageSegmenter.create_from_options(options)
```

Por último, la forma de procesar los frames a partir de la segmentación es la siguiente:

```
# Creamos el FaceMesh y procesamos los frames
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertimos el frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Obtenemos los resultados del segmentador
        segmenter.segment_async(frame_rgb, time.time_ns() // 1_000_000)
```

Este código muestra cómo procesar los frames tras la segmentación, utilizando la conversión de los mismos a formato RGB y luego pasándolos al segmentador para obtener los resultados de forma asíncrona.





---

> [!IMPORTANT]  
> Los archivos presentados aquí son una modificación de los archivos originales de [otsedom](https://github.com/otsedom/otsedom.github.io/tree/main/VC).



---

## 📚 Bibliografía

1. [Mediapipe](https://github.com/google-ai-edge/mediapipe)
2. [Image Segmenter - Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter?hl=es-419)
3. [Detección de rostros con Mediapipe y Python](https://omes-va.com/deteccion-de-rostros-mediapipe-python/)
4. [Selfie Segmentation con Mediapipe y Python](https://omes-va.com/mediapipe-selfie-segmentation-python-2/)
5. [Malla facial con Mediapipe y Python](https://omes-va.com/malla-facial-mediapipe-python/)

---

**Universidad de Las Palmas de Gran Canaria**  

EII - Grado de Ingeniería Informática  
Obra bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional

---
