import cv2
import logging
from ultralytics import YOLO
import numpy as np
import os
from pydantic import BaseModel, Field, model_validator

LOG_FILENAME = '../detections.log'
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

MAX_BLUR_LEVEL = 10
MIN_BLUR_LEVEL = 0
DEFAULT_BLUR_LEVEL = 5


class BackgroundObfuscationConfig(BaseModel):
    """
    Modelo de configuración para la clase BackgroundObfuscation.

    Atributos:
        yolo_model_path (str): Ruta al modelo YOLO preentrenado.
        confidence_thresholds (dict): Diccionario de umbrales de confianza específicos para cada clase.
        blur_background (bool): Si True, se desenfoca el fondo. Si False, se usa una imagen o video como fondo.
        blur_level (int): Nivel de desenfoque de 0 a 10.
        background_image_path (str): Ruta a la imagen de fondo si blur_background es False.
        background_video_path (str): Ruta al video de fondo si blur_background es False y se quiere un video como fondo.
        detect_phones (bool): Si True, se detectan teléfonos, si False, se ignoran las detecciones de teléfonos.
    """
    yolo_model_path: str = '../yolo11n-seg.pt'
    confidence_thresholds: dict = Field({
        0: 0.7,  # Umbral de confianza para personas (ID de clase 0)
        67: 0.8,  # Umbral de confianza para teléfonos (ID de clase 67)
        73: 0.8  # Umbral de confianza para cámaras (ID de clase 73)
    })
    blur_background: bool = True
    blur_level: int = Field(DEFAULT_BLUR_LEVEL, ge=MIN_BLUR_LEVEL, le=MAX_BLUR_LEVEL)
    background_image_path: str = None
    background_video_path: str = None
    detect_phones: bool = True

    model_config = {'protected_namespaces': ()}

    @model_validator(mode="before")
    def check_paths(cls, values):
        """
        Valida que las rutas para el modelo YOLO, imagen o video de fondo existan.
        """
        yolo_model_path = values.get('yolo_model_path')
        background_image_path = values.get('background_image_path')
        background_video_path = values.get('background_video_path')
        blur_background = values.get('blur_background')

        # Validar la existencia de la ruta del modelo YOLO
        if not os.path.exists(yolo_model_path):
            logger.error(f"Modelo no encontrado en la ruta especificada: {yolo_model_path}")
            raise FileNotFoundError(f"Modelo no encontrado en la ruta especificada: {yolo_model_path}")

        # Validar la imagen o el video de fondo solo si no se desenfoca el fondo
        if not blur_background:
            if not background_image_path and not background_video_path:
                raise ValueError("Debe proporcionar una ruta de imagen o video de fondo si 'blur_background' es False.")

            if background_image_path and not os.path.exists(background_image_path):
                raise FileNotFoundError(f"Imagen de fondo no encontrada en la ruta "
                                        f"especificada: {background_image_path}")

            if background_video_path and not os.path.exists(background_video_path):
                raise FileNotFoundError(f"Video de fondo no encontrado en "
                                        f"la ruta especificada: {background_video_path}")

        return values


class BackgroundObfuscation:
    """
    Clase para aplicar desenfoque o cambio de fondo en función de detecciones de personas utilizando YOLO.

    Atributos:
        model (YOLO): Modelo YOLO cargado para detección.
        confidence_thresholds (dict): Diccionario de umbrales de confianza específicos para cada clase.
        blur_background (bool): Si True, se desenfoca el fondo. Si False, se usa una imagen o video como fondo.
        blur_level (int): Nivel de desenfoque de 0 a 10.
        background_image (ndarray): Imagen de fondo si se utiliza cambio de fondo en lugar de desenfoque.
        background_video (cv2.VideoCapture): Video de fondo si se utiliza cambio de fondo con un video.
        detect_phones (bool): Si True, se detectan teléfonos, si False, se ignoran las detecciones de teléfonos.
    """

    def __init__(self, config: BackgroundObfuscationConfig):
        """
        Inicializa la clase BackgroundObfuscation con la configuración especificada.

        Args:
            config (BackgroundObfuscationConfig): Configuración para la clase.
        """
        self.model = self.load_model(config.yolo_model_path)
        self.confidence_thresholds = config.confidence_thresholds
        self.blur_background = config.blur_background
        self.blur_level = config.blur_level
        self.background_image = None
        self.background_video = None
        self.detect_phones = config.detect_phones  # Nueva variable para detección de teléfonos

        if not self.blur_background:
            if config.background_image_path:
                self.background_image = self.load_background_image(config.background_image_path)
            if config.background_video_path:
                self.background_video = self.load_background_video(config.background_video_path)

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
            logger.info(f"Modelo YOLOv11 cargado exitosamente desde {model_path}")
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

    @staticmethod
    def load_background_video(background_video_path: str) -> cv2.VideoCapture:
        """
        Carga el video de fondo desde la ruta especificada.

        Args:
            background_video_path (str): Ruta al video de fondo.

        Returns:
            cv2.VideoCapture: Objeto de captura de video de fondo.
        """
        background_video = cv2.VideoCapture(background_video_path)
        if not background_video.isOpened():
            logger.error(f"No se pudo abrir el video de fondo desde {background_video_path}")
            raise ValueError(f"No se pudo abrir el video de fondo desde {background_video_path}")
        logger.info(f"Video de fondo cargado desde {background_video_path}")
        return background_video

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
            threshold = self.confidence_thresholds.get(class_id, 0.5)

            if class_id == 67 and not self.detect_phones:  # Si detect_phones es False, ignoramos teléfonos
                continue

            if class_id in restricted_classes and confidence >= threshold:
                logger.info(
                    f"Objeto restringido detectado: clase {class_id}, "
                    f"confianza {confidence:.2f}, umbral {threshold:.2f}")
                return True
        return False

    def generate_warning_message(self, frame: np.ndarray) -> np.ndarray:
        """
        Genera un mensaje de advertencia en el fotograma.

        Args:
            frame (ndarray): Fotograma original.

        Returns:
            ndarray: Fotograma con mensaje de advertencia.
        """
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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un fotograma para desenfocar el fondo o cambiarlo con imagen/video,
        mientras mantiene a las personas visibles.

        Args:
            frame (ndarray): Fotograma capturado por la cámara o video.

        Returns:
            ndarray: Fotograma con el fondo desenfocado o cambiado y las personas visibles.
        """
        results = self.model(frame)
        if self.detect_restricted_items(results) and self.detect_phones:
            return self.generate_warning_message(frame)

        mask_person = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        if results[0].masks is not None:
            masks = results[0].masks.data
            for i in range(len(masks)):
                class_id = results[0].boxes.cls[i].item()
                confidence = results[0].boxes.conf[i].item()
                threshold = self.confidence_thresholds.get(0, self.confidence_thresholds[0])
                if class_id == 0 and confidence >= threshold:  # Clase 'person'
                    mask = masks[i].numpy().astype(np.uint8)
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_person[mask_resized > 0] = 255
        mask_background = cv2.bitwise_not(mask_person)

        if self.blur_background:
            return self.apply_blur(frame, mask_person, mask_background)
        else:
            return self.apply_background_replacement(frame, mask_person, mask_background)

    def apply_blur(self, frame: np.ndarray, mask_person: np.ndarray, mask_background: np.ndarray) -> np.ndarray:
        """
        Aplica desenfoque al fondo de la imagen.

        Args:
            frame (ndarray): Fotograma original.
            mask_person (ndarray): Máscara de la persona.
            mask_background (ndarray): Máscara del fondo.

        Returns:
            ndarray: Fotograma con fondo desenfocado.
        """
        kernel_size = (5 + 2 * self.blur_level, 5 + 2 * self.blur_level)
        sigma = 10 + self.blur_level * 7
        background_blurred = cv2.GaussianBlur(frame, kernel_size, sigma)
        person_only = cv2.bitwise_and(frame, frame, mask=mask_person)
        background_only = cv2.bitwise_and(background_blurred, background_blurred, mask=mask_background)
        return cv2.add(person_only, background_only)

    def apply_background_replacement(self, frame: np.ndarray, mask_person: np.ndarray,
                                     mask_background: np.ndarray) -> np.ndarray:
        """
        Reemplaza el fondo de la imagen con una imagen o video.

        Args:
            frame (ndarray): Fotograma original.
            mask_person (ndarray): Máscara de la persona.
            mask_background (ndarray): Máscara del fondo.

        Returns:
            ndarray: Fotograma con el fondo reemplazado.
        """
        background_resized = None
        if self.background_image is not None:
            background_resized = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))
        elif self.background_video is not None:
            ret, background_frame = self.background_video.read()
            if not ret:
                self.background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Repetir el video
                ret, background_frame = self.background_video.read()
            background_resized = cv2.resize(background_frame, (frame.shape[1], frame.shape[0]))

        if background_resized is not None:
            person_only = cv2.bitwise_and(frame, frame, mask=mask_person)
            background_only = cv2.bitwise_and(background_resized, background_resized, mask=mask_background)
            return cv2.add(person_only, background_only)
        return frame

    def process_image(self, image_path: str) -> None:
        """
        Procesa una imagen individual, guarda el resultado y lo muestra en pantalla.

        Args:
            image_path (str): Ruta de la imagen a procesar.
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
        Procesa un archivo de video y guarda el resultado, procesando cada 15 fotogramas para mayor fluidez.

        Args:
            video_path (str): Ruta del video a procesar.
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

        frame_count = 0  # Contador de fotogramas

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Procesamiento de Video', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Procesamiento de video terminado por el usuario")
                break
            frame_count += 1
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Video procesado guardado en {output_path}")

    def process_camera(self, camera_index: int = 0) -> None:
        """
        Procesa video en tiempo real desde la cámara.

        Args:
            camera_index (int): Índice de la cámara. 0 para la cámara por defecto.
        """
        try:
            cap = self.initialize_camera(camera_index)
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Error: No se puede recibir fotogramas")
                    break

                frame_with_background = self.process_frame(frame)
                cv2.imshow('Detección de Segmentos con Fondo Aplicado - YOLOv11-Seg', frame_with_background)

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
    yolo_model_path='../yolo11l-seg.pt',
    confidence_thresholds={
        0: 0.6,  # Umbral para personas
        67: 0.3,  # Umbral para teléfonos
        73: 0.4  # Umbral para cámaras
    },
    blur_background=True,
    blur_level=10,
    background_image_path='../ejercicio_1/img_1.png',  # Imagen de fondo
    background_video_path='../ejercicio_1/fondo.mp4',  # Video de fondo
    detect_phones=False  # Detección de teléfonos habilitada
)

background_obfuscation = BackgroundObfuscation(config)
# background_obfuscation.run(mode='camera')
# background_obfuscation.run(mode='image', source='ejercicio_1/img.png')
background_obfuscation.run(mode='video', source='ejercicio_1/video1.mp4')
