import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
import numpy as np
import logging
from abc import ABC, abstractmethod

# Configurar el logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Interfaz para la estrategia de cálculo de peso de frutas
class FruitStrategy(ABC):
    @abstractmethod
    def calcular_peso(self, mask, escala_pix_cm2):
        pass

    @abstractmethod
    def obtener_diametro_real(self):
        pass

# Estrategia para manzana
class AppleStrategy(FruitStrategy):
    densidad = 0.9
    diametro_promedio = 8

    def calcular_peso(self, mask, escala_pix_cm2):
        area_pixeles = np.sum(mask)
        area_cm2 = area_pixeles * escala_pix_cm2
        volumen_aprox = area_cm2 * AppleStrategy.diametro_promedio
        peso_estimado = AppleStrategy.densidad * volumen_aprox
        return peso_estimado

    def obtener_diametro_real(self):
        return AppleStrategy.diametro_promedio

# Estrategia para banana
class BananaStrategy(FruitStrategy):
    densidad = 0.94
    longitud_promedio = 20
    diametro_promedio = 3.5

    def calcular_peso(self, mask, escala_pix_cm2):
        area_pixeles = np.sum(mask)
        area_cm2 = area_pixeles * escala_pix_cm2
        volumen_aprox = area_cm2 * BananaStrategy.longitud_promedio
        peso_estimado = BananaStrategy.densidad * volumen_aprox
        return peso_estimado

    def obtener_diametro_real(self):
        return BananaStrategy.diametro_promedio

# Estrategia para naranja
class OrangeStrategy(FruitStrategy):
    densidad = 0.96  # Densidad promedio de una naranja en g/cm³
    diametro_promedio = 7.5  # Diámetro promedio en cm

    def calcular_peso(self, mask, escala_pix_cm2):
        area_pixeles = np.sum(mask)
        area_cm2 = area_pixeles * escala_pix_cm2
        volumen_aprox = area_cm2 * OrangeStrategy.diametro_promedio
        peso_estimado = OrangeStrategy.densidad * volumen_aprox
        return peso_estimado

    def obtener_diametro_real(self):
        return OrangeStrategy.diametro_promedio

# Estrategia para pera
class PearStrategy(FruitStrategy):
    densidad = 0.6  # Densidad promedio de una pera en g/cm³
    diametro_promedio = 6  # Diámetro promedio en cm

    def calcular_peso(self, mask, escala_pix_cm2):
        area_pixeles = np.sum(mask)
        area_cm2 = area_pixeles * escala_pix_cm2
        volumen_aprox = area_cm2 * PearStrategy.diametro_promedio
        peso_estimado = PearStrategy.densidad * volumen_aprox
        return peso_estimado

    def obtener_diametro_real(self):
        return PearStrategy.diametro_promedio

# Contenedor de estrategias de frutas
class FruitWeightEstimator:
    def __init__(self, estrategias):
        # Configurar Detectron2
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Ajustar el umbral de confianza a 0.5
        self.cfg.MODEL.DEVICE = 'cpu'  # Usar CPU si no hay GPU disponible

        # Crear el predictor
        self.predictor = DefaultPredictor(self.cfg)

        # Estrategias de frutas
        self.estrategias = estrategias

    def calcular_escala_automatica(self, diametro_promedio_px, diametro_real_cm):
        """
        Calcula la escala de píxeles a cm² basándose en el diámetro promedio en píxeles.
        """
        return (diametro_real_cm ** 2) / (np.pi * (diametro_promedio_px / 2) ** 2)

    def process_frame(self, frame):
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes
        pred_masks = instances.pred_masks

        for i, class_id in enumerate(pred_classes):
            class_name = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[class_id]
            # Buscar estrategia correspondiente a la fruta detectada
            estrategia = self.estrategias.get(class_name.lower())

            if estrategia:
                mask = pred_masks[i].numpy()
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                diametro_promedio_px = (w + h) / 2
                escala_pix_cm2 = self.calcular_escala_automatica(diametro_promedio_px, estrategia.obtener_diametro_real())
                peso_estimado = estrategia.calcular_peso(mask, escala_pix_cm2)
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                texto = f'{class_name.capitalize()}: {peso_estimado:.2f} g'
                cv2.putText(frame, texto, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def process_image(self, image_path):
        """
        Procesa una imagen para estimar el peso de las frutas detectadas.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen en la ruta {image_path}. Verifica la ruta del archivo.")
            return
        processed_frame = self.process_frame(image)
        cv2.imshow('Detecciones y Peso Estimado', processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path):
        """
        Procesa un video para estimar el peso de las frutas detectadas en cada frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video en la ruta {video_path}. Verifica la ruta del archivo.")
            return
        print("Procesando video...")  # Mensaje para verificar si se está iniciando el proceso
        frame_index = 0
        cv2.namedWindow('Detecciones y Peso Estimado', cv2.WINDOW_NORMAL)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer el frame.")
                break
            if frame_index % 25 == 0:
                print(f"Procesando frame {frame_index}...")
                frame_resized = cv2.resize(frame, (640, 360))
                processed_frame = self.process_frame(frame_resized)
                cv2.imshow('Detecciones y Peso Estimado', processed_frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
            frame_index += 1
        cap.release()
        cv2.destroyAllWindows()


# Crear instancias de estrategias y contenedor
estrategias_frutas = {
    'apple': AppleStrategy(),
    'banana': BananaStrategy(),
    'orange': OrangeStrategy(),
    'pear': PearStrategy()
}

# Crear instancia del estimador con estrategias para múltiples frutas
estimator = FruitWeightEstimator(estrategias_frutas)
# Procesar una imagen o un video
# estimator.process_image('img_4.png')  # Procesar una imagen
# estimator.process_image('img_5.png')  # Procesar una imagen

#estimator.process_video("manzanas.mp4")  # Procesar video con múltiples frutas
#estimator.process_video("banana2.mp4")  # Procesar video con múltiples frutas
#estimator.process_video('peras.mp4')
estimator.process_video('mixto_3.mp4')
