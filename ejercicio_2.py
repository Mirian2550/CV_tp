import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
from abc import ABC, abstractmethod


class FruitStrategy(ABC):
    """
    Interfaz para definir estrategias de cálculo de peso de frutas.
    """

    @abstractmethod
    def calcular_peso(self, mask, escala_pix_cm2):
        """
        Calcula el peso de la fruta a partir de la máscara de segmentación y la escala de píxeles a cm².

        :param mask: Máscara binaria de la fruta.
        :param escala_pix_cm2: Escala de píxeles a centímetros cuadrados.
        :return: Peso estimado en gramos.
        """
        pass

    @abstractmethod
    def obtener_diametro_real(self):
        """
        Obtiene el diámetro promedio de la fruta en cm.

        :return: Diámetro promedio en cm.
        """
        pass


class AppleStrategy(FruitStrategy):
    """
    Estrategia de cálculo de peso para una manzana, aproximada a una esfera.
    """
    densidad = 0.9
    diametro_promedio = 8
    peso_min = 100
    peso_max = 230

    def calcular_peso(self, mask, escala_pix_cm2):
        area_pixeles = np.sum(mask)
        area_cm2 = area_pixeles * escala_pix_cm2
        radio_cm = np.sqrt(area_cm2 / np.pi)
        volumen_aprox = (4 / 3) * np.pi * (radio_cm ** 3)
        peso_estimado = AppleStrategy.densidad * volumen_aprox
        return max(AppleStrategy.peso_min, min(peso_estimado, AppleStrategy.peso_max))

    def obtener_diametro_real(self):
        return AppleStrategy.diametro_promedio


class BananaStrategy(FruitStrategy):
    """
    Estrategia de cálculo de peso para una banana, aproximada a un cilindro.
    """
    densidad = 0.94
    longitud_promedio = 20
    diametro_promedio = 3.5
    peso_min = 80
    peso_max = 200

    def calcular_peso(self, mask, escala_pix_cm2):
        area_pixeles = np.sum(mask)
        area_cm2 = area_pixeles * escala_pix_cm2
        radio_cm = self.diametro_promedio / 2
        volumen_aprox = np.pi * (radio_cm ** 2) * BananaStrategy.longitud_promedio
        peso_estimado = BananaStrategy.densidad * volumen_aprox
        return max(BananaStrategy.peso_min, min(peso_estimado, BananaStrategy.peso_max))

    def obtener_diametro_real(self):
        return BananaStrategy.diametro_promedio


class OrangeStrategy(FruitStrategy):
    """
    Estrategia de cálculo de peso para una naranja, aproximada a una esfera.
    """
    densidad = 0.96
    diametro_promedio = 7.5
    peso_min = 120
    peso_max = 210

    def calcular_peso(self, mask, escala_pix_cm2):
        area_pixeles = np.sum(mask)
        area_cm2 = area_pixeles * escala_pix_cm2
        radio_cm = np.sqrt(area_cm2 / np.pi)
        volumen_aprox = (4 / 3) * np.pi * (radio_cm ** 3)
        peso_estimado = OrangeStrategy.densidad * volumen_aprox
        return max(OrangeStrategy.peso_min, min(peso_estimado, OrangeStrategy.peso_max))

    def obtener_diametro_real(self):
        return OrangeStrategy.diametro_promedio


class PearStrategy(FruitStrategy):
    """
    Estrategia de cálculo de peso para una pera, aproximada a una esfera con menor densidad.
    """
    densidad = 0.6
    diametro_promedio = 6
    peso_min = 100
    peso_max = 160

    def calcular_peso(self, mask, escala_pix_cm2):
        area_pixeles = np.sum(mask)
        area_cm2 = area_pixeles * escala_pix_cm2
        radio_cm = np.sqrt(area_cm2 / np.pi)
        volumen_aprox = (4 / 3) * np.pi * (radio_cm ** 3)
        peso_estimado = PearStrategy.densidad * volumen_aprox
        return max(PearStrategy.peso_min, min(peso_estimado, PearStrategy.peso_max))

    def obtener_diametro_real(self):
        return PearStrategy.diametro_promedio


class SegmentationProcessor:
    """
    Procesador de segmentación para calcular el peso de frutas utilizando un modelo preentrenado.
    """

    def __init__(self, model_path, class_names, thresholds_per_class, delay=2000, estrategias=None, device='cpu'):
        """
        Inicializa el procesador con la configuración del modelo y las estrategias de frutas.

        :param model_path: Ruta al modelo entrenado.
        :param class_names: Lista de nombres de las clases de frutas.
        :param thresholds_per_class: Umbrales por clase para las detecciones.
        :param delay: Tiempo de retención de la predicción en milisegundos.
        :param estrategias: Diccionario de estrategias de frutas.
        :param device: Dispositivo a utilizar ('cpu' o 'cuda').
        """
        self.class_names = class_names
        self.thresholds_per_class = thresholds_per_class
        self.delay = delay
        self.estrategias = estrategias

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)

    def calcular_escala_automatica(self, diametro_promedio_px, diametro_real_cm):
        """
        Calcula la escala de píxeles a cm² en base al diámetro promedio en píxeles.

        :param diametro_promedio_px: Diámetro promedio en píxeles.
        :param diametro_real_cm: Diámetro real en centímetros.
        :return: Escala de píxeles a cm².
        """
        return (diametro_real_cm ** 2) / (np.pi * (diametro_promedio_px / 2) ** 2)

    def process_frame(self, frame):
        """
        Procesa un fotograma para realizar la segmentación y cálculo del peso de las frutas.

        :param frame: Imagen de entrada.
        :return: Imagen con la segmentación y el peso calculado.
        """
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes if instances.has("pred_classes") else None
        pred_masks = instances.pred_masks if instances.has("pred_masks") else None
        scores = instances.scores if instances.has("scores") else None

        mask_image = np.zeros_like(frame)

        if pred_masks is not None:
            for i, mask in enumerate(pred_masks):
                class_id = pred_classes[i].item()
                class_name = self.class_names[class_id]
                class_threshold = self.thresholds_per_class[class_name]

                if scores[i] > class_threshold:
                    mask = mask.numpy()
                    mask_image[mask] = [0, 255, 0]

                    estrategia = self.estrategias.get(class_name.lower())
                    if estrategia:
                        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        diametro_promedio_px = (w + h) / 2
                        escala_pix_cm2 = self.calcular_escala_automatica(diametro_promedio_px,
                                                                         estrategia.obtener_diametro_real())
                        peso_estimado = estrategia.calcular_peso(mask, escala_pix_cm2)

                        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                        texto = f'{class_name.capitalize()}: {peso_estimado:.2f} g'
                        cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        combined_image = cv2.addWeighted(frame, 1, mask_image, 0.5, 0)
        return combined_image

    def process_image(self, image_path):
        """
        Procesa una imagen estática.

        :param image_path: Ruta de la imagen a procesar.
        """
        image = cv2.imread(image_path)
        result = self.process_frame(image)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    def process_video(self, video_path):
        """
        Procesa un archivo de video realizando detecciones cada 60 fotogramas.

        :param video_path: Ruta del archivo de video.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 60 == 0:
                result = self.process_frame(frame)
                cv2.imshow("Segmentación en Video", result)
                cv2.waitKey(self.delay)
            else:
                cv2.imshow("Segmentación en Video", frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_camera(self, camera_index=0):
        """
        Procesa el feed de una cámara en tiempo real y realiza detecciones cada 60 fotogramas.

        :param camera_index: Índice de la cámara (0 por defecto).
        """
        cap = cv2.VideoCapture(camera_index)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 60 == 0:
                result = self.process_frame(frame)
                cv2.imshow("Segmentación en Cámara", result)
                cv2.waitKey(self.delay)
            else:
                cv2.imshow("Segmentación en Cámara", frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Uso de la clase

class_names = ['apple', 'banana', 'orange', 'pear']
thresholds_per_class = {
    'apple': 0.85,
    'banana': 0.80,
    'orange': 0.85,
    'pear': 0.15
}

estrategias_frutas = {
    'apple': AppleStrategy(),
    'banana': BananaStrategy(),
    'orange': OrangeStrategy(),
    'pear': PearStrategy()
}

model_path = "./modelos/segmentacion/model_final.pth"

processor = SegmentationProcessor(model_path, class_names, thresholds_per_class, delay=2000,
                                  estrategias=estrategias_frutas)

#processor.process_image("fotos_frutas/8.jpeg")
#processor.process_video("videos_frutas/video_1.mp4")
processor.process_camera()