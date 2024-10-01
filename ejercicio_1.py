import cv2
import logging
from ultralytics import YOLO
import numpy as np
import os
import requests
from pydantic import BaseModel, Field, field_validator

LOG_FILENAME = 'detections.log'
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

MAX_BLUR_LEVEL = 10
MIN_BLUR_LEVEL = 0
DEFAULT_BLUR_LEVEL = 5

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"


class BackgroundObfuscationConfig(BaseModel):
    """
    Modelo de configuración para la clase BackgroundObfuscation.

    Atributos:
        model_path (str): Ruta al modelo YOLO preentrenado.
        confidence_thresholds (dict): Diccionario de umbrales de confianza específicos para cada clase.
        blur_background (bool): Si True, se desenfoca el fondo. Si False, se usa una imagen como fondo.
        blur_level (int): Nivel de desenfoque de 0 a 10, donde 0 es sin desenfoque y 10 es el máximo desenfoque.
        background_image_path (str): Ruta a la imagen de fondo si blur_background es False.
    """
    model_path: str = 'yolov8n-seg.pt'
    confidence_thresholds: dict = Field({
        0: 0.7,  # Umbral de confianza para personas (ID de clase 0)
        67: 0.8,  # Umbral de confianza para teléfonos (ID de clase 67)
        73: 0.8  # Umbral de confianza para cámaras (ID de clase 73)
    })
    blur_background: bool = True
    blur_level: int = Field(DEFAULT_BLUR_LEVEL, ge=MIN_BLUR_LEVEL, le=MAX_BLUR_LEVEL)
    background_image_path: str = None

    model_config = {
        'protected_namespaces': ()
    }

    @field_validator('model_path')
    def validate_model_path(cls, v):
        if not os.path.exists(v):
            logger.warning(f"Modelo no encontrado en la ruta especificada: {v}")
            try:
                cls.download_model(v)
            except Exception as e:
                raise FileNotFoundError(f"No se pudo descargar el modelo: {e}")
        return v

    @staticmethod
    def download_model(model_path: str):
        logger.info(f"Descargando el modelo desde {MODEL_URL} ...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Modelo descargado y guardado en {model_path}")
        except Exception as e:
            logger.error(f"Error al descargar el modelo: {e}")
            raise

    @field_validator('background_image_path')
    def validate_background_image_path(cls, v, info):
        blur_background = info.data['blur_background']
        if not blur_background and not v:
            raise ValueError("Debe proporcionar una ruta de imagen de fondo si 'blur_background' es False.")
        if v and not os.path.exists(v):
            raise FileNotFoundError(f"Imagen de fondo no encontrada en la ruta especificada: {v}")
        return v


class BackgroundObfuscation:
    """
    Clase para aplicar desenfoque o cambio de fondo en función de detecciones de personas utilizando YOLO.

    Atributos:
        model (YOLO): Modelo YOLO cargado para detección.
        confidence_thresholds (dict): Diccionario de umbrales de confianza específicos para cada clase.
        blur_background (bool): Si True, se desenfoca el fondo. Si False, se usa una imagen como fondo.
        blur_level (int): Nivel de desenfoque de 0 a 10.
        background_image (ndarray): Imagen de fondo si se utiliza cambio de fondo en lugar de desenfoque.
    """

    def __init__(self, config: BackgroundObfuscationConfig):
        """
        Inicializa la clase BackgroundObfuscation con la configuración especificada.

        Args:
            config (BackgroundObfuscationConfig): Configuración para la clase.
        """
        self.model = self.load_model(config.model_path)
        self.confidence_thresholds = config.confidence_thresholds
        self.blur_background = config.blur_background
        self.blur_level = config.blur_level
        self.background_image = self.load_background_image(
            config.background_image_path) if config.background_image_path else None

    @staticmethod
    def load_model(model_path: str) -> YOLO:
        """
        Carga el modelo YOLO desde la ruta especificada.

        Args:
            model_path (str): Ruta al modelo YOLO.

        Returns:
            YOLO: Modelo YOLO cargado.

        Raises:
            Exception: Si ocurre un error al cargar el modelo.
        """
        try:
            model = YOLO(model_path)
            logger.info(f"Modelo YOLO cargado exitosamente desde {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise

    @staticmethod
    def load_background_image(background_image_path: str) -> np.ndarray:
        """
        Carga la imagen de fondo desde la ruta especificada.

        Args:
            background_image_path (str): Ruta a la imagen de fondo.

        Returns:
            ndarray: Imagen de fondo cargada.

        Raises:
            ValueError: Si no se puede cargar la imagen de fondo.
        """
        background_image = cv2.imread(background_image_path)
        if background_image is None:
            logger.error(f"No se pudo cargar la imagen de fondo desde {background_image_path}")
            raise ValueError(f"No se pudo cargar la imagen de fondo desde {background_image_path}")
        logger.info(f"Imagen de fondo cargada desde {background_image_path}")
        return background_image

    def detect_restricted_items(self, results) -> bool:
        """
        Detecta la presencia de objetos restringidos como teléfonos o cámaras en los resultados.

        Args:
            results: Resultados de la detección YOLO.

        Returns:
            bool: True si se detecta un objeto restringido, False en caso contrario.
        """
        restricted_classes = [67, 73]  # IDs de clases para teléfono y cámara en el dataset COCO
        for i in range(len(results[0].boxes)):
            class_id = results[0].boxes.cls[i].item()
            confidence = results[0].boxes.conf[i].item()
            threshold = self.confidence_thresholds.get(class_id, 0.5)  # Umbral por defecto de 0.5 si no se especifica
            if class_id in restricted_classes and confidence >= threshold:
                logger.info(
                    f"Objeto restringido detectado: "
                    f"clase {class_id}, "
                    f"confianza {confidence:.2f}, "
                    f"umbral {threshold:.2f}")
                return True
        return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un fotograma para desenfocar el fondo o cambiarlo, mientras mantiene a las personas visibles.

        Args:
            frame (ndarray): Fotograma capturado por la cámara o video.

        Returns:
            ndarray: Fotograma con el fondo desenfocado o cambiado y las personas visibles.
        """
        results = self.model(frame)

        if self.detect_restricted_items(results):
            frame_black = np.zeros_like(frame)
            message_lines = ["NO ME GRABES ...", "JAAJAJ"]
            font_scale = 2
            font_thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX

            line_height = cv2.getTextSize("Tg", font, font_scale, font_thickness)[0][1]

            total_text_height = len(message_lines) * line_height
            text_y = (frame.shape[0] - total_text_height) // 2

            for i, line in enumerate(message_lines):
                text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                cv2.putText(frame_black, line, (text_x, text_y + i * line_height), font, font_scale, (255, 255, 255),
                            font_thickness)

            return frame_black

        mask_person = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        if results[0].masks is not None:
            masks = results[0].masks.data
            for i in range(len(masks)):
                class_id = results[0].boxes.cls[i].item()
                confidence = results[0].boxes.conf[i].item()
                threshold = self.confidence_thresholds.get(0, self.confidence_thresholds[0])
                if class_id == 0 and confidence >= threshold:  # Clase 'person'
                    mask = masks[i].numpy().astype(np.uint8)
                    mask_resized = cv2.resize(mask,
                                              (frame.shape[1], frame.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                    mask_person[mask_resized > 0] = 255
        mask_background = cv2.bitwise_not(mask_person)

        if self.blur_background:
            kernel_size = (5 + 2 * self.blur_level, 5 + 2 * self.blur_level)
            sigma = 10 + self.blur_level * 7
            background_blurred = cv2.GaussianBlur(frame, kernel_size, sigma)
            person_only = cv2.bitwise_and(frame, frame, mask=mask_person)
            background_only = cv2.bitwise_and(background_blurred, background_blurred, mask=mask_background)
        else:
            background_resized = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))
            person_only = cv2.bitwise_and(frame, frame, mask=mask_person)
            background_only = cv2.bitwise_and(background_resized, background_resized, mask=mask_background)

        combined_frame = cv2.add(person_only, background_only)

        return combined_frame

    def process_image(self, image_path: str) -> None:
        """
        Procesa una imagen individual, guarda el resultado y lo muestra en pantalla.

        Args:
            image_path (str): Ruta de la imagen a procesar.

        Raises:
            FileNotFoundError: Si la imagen no se encuentra en la ruta especificada.
            ValueError: Si la imagen no se puede cargar.
        """
        if not os.path.exists(image_path):
            logger.error(f"Imagen no encontrada: {image_path}")
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"No se pudo cargar la imagen: {image_path}")
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        processed_image = self.process_frame(image)
        output_path = f"processed_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, processed_image)
        logger.info(f"Imagen procesada guardada en {output_path}")

        # Mostrar la imagen procesada
        cv2.imshow('Imagen Procesada', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path: str) -> None:
        """
        Procesa un archivo de video y guarda el resultado.

        Args:
            video_path (str): Ruta del video a procesar.

        Raises:
            FileNotFoundError: Si el video no se encuentra en la ruta especificada.
            ValueError: Si el video no se puede cargar.
        """
        if not os.path.exists(video_path):
            logger.error(f"Video no encontrado: {video_path}")
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {video_path}")
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f"processed_{os.path.basename(video_path)}"
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            out.write(processed_frame)

            # Mostrar el video en tiempo real mientras se procesa
            cv2.imshow('Procesamiento de Video - YOLOv8-Seg', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Procesamiento de video terminado por el usuario")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Video procesado guardado en {output_path}")

    def process_camera(self, camera_index: int = 0) -> None:
        """
        Procesa video en tiempo real desde la cámara.

        Args:
            camera_index (int): Índice de la cámara. 0 para la cámara por defecto.

        Raises:
            Exception: Si no se puede inicializar la cámara.
        """
        try:
            cap = self.initialize_camera(camera_index)
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error: No se puede recibir fotogramas")
                    break

                frame_with_background = self.process_frame(frame)
                cv2.imshow('Detección de Segmentos con Fondo Aplicado - YOLOv8-Seg', frame_with_background)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Detección terminada por el usuario")
                    break

        except Exception as e:
            logger.error(f"Error en el proceso de detección: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Captura de video finalizada")

    @staticmethod
    def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
        """
        Inicializa la captura de video desde la cámara.

        Args:
            camera_index (int): Índice de la cámara. 0 para la cámara por defecto.

        Returns:
            cv2.VideoCapture: Objeto de captura de video.

        Raises:
            Exception: Si no se puede abrir la cámara.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Error: No se puede abrir la cámara")
            raise Exception("No se puede abrir la cámara")
        logger.info("Captura de video iniciada")
        return cap

    def run(self, mode: str = 'camera', source: str = None) -> None:
        """
        Ejecuta el procesamiento en el modo especificado.

        Args:
            mode (str): Modo de ejecución ('camera', 'image', 'video').
            source (str): Fuente del archivo de imagen o video, si aplica.

        Raises:
            ValueError: Si el modo o la fuente no son válidos.
        """
        if mode == 'camera':
            self.process_camera()
        elif mode == 'image' and source:
            self.process_image(source)
        elif mode == 'video' and source:
            self.process_video(source)
        else:
            logger.error("Modo o fuente inválidos. Modo debe ser 'camera', "
                         "'image' o 'video' y fuente no puede ser None para 'image' o 'video'.")
            raise ValueError("Modo o fuente inválidos. Modo debe ser 'camera', "
                             "'image' o 'video' y fuente no puede ser None para 'image' o 'video'.")


# Configuración inicial y ejecución
config = BackgroundObfuscationConfig(
    model_path='yolov8x-seg.pt',
    confidence_thresholds={
        0: 0.6,  # Umbral para personas
        67: 0.3,  # Umbral para teléfonos
        73: 0.4  # Umbral para cámaras
    },
    blur_background=True,
    blur_level=10,
    background_image_path='img_1.png'
)

background_obfuscation = BackgroundObfuscation(config)
background_obfuscation.run(mode='camera')
# background_obfuscation.run(mode='image', source='img.png')
#background_obfuscation.run(mode='video', source='video.mp4')
