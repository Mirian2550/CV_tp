import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Inicializar ORB y BFMatcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

class FruitStrategy(ABC):
    @abstractmethod
    def contar_frutas(self, mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current):
        pass

    def contar_y_marcar(self, fruit_name, mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        centro = (x + w // 2, y + h // 2)
        area = cv2.contourArea(largest_contour)
        if not any(np.linalg.norm(np.array(centro) - np.array(pos)) < 50 for pos, _, _ in detected_positions[fruit_name]):
            coincidencia = False
            for (prev_center, prev_keypoints, prev_descriptors) in detected_positions[fruit_name]:
                matches = bf.match(descriptors_current, prev_descriptors)
                good_matches = [m for m in matches if m.distance < 30]
                if len(good_matches) > 5:
                    coincidencia = True
                    break
            if not coincidencia:
                detected_fruits[fruit_name] += 1
                detected_positions[fruit_name].append((centro, keypoints_current, descriptors_current))
                frame[mask > 0] = (0, 255, 0)
        else:
            frame[mask > 0] = (128, 128, 128)

class AppleStrategy(FruitStrategy):
    def contar_frutas(self, mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current):
        self.contar_y_marcar('manzana', mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current)

class BananaStrategy(FruitStrategy):
    def contar_frutas(self, mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current):
        self.contar_y_marcar('banana', mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current)

class OrangeStrategy(FruitStrategy):
    def contar_frutas(self, mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current):
        self.contar_y_marcar('naranja', mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current)

class PearStrategy(FruitStrategy):
    def contar_frutas(self, mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current):
        self.contar_y_marcar('pera', mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current)

# Registrar la metadata de tu dataset personalizado
from detectron2.data import MetadataCatalog

# Definir tus clases personalizadas
custom_classes = ['manzana', 'banana', 'naranja', 'pera']

# Registrar la metadata para tu dataset
fruit_metadata = MetadataCatalog.get("fruit_dataset")
fruit_metadata.set(thing_classes=custom_classes)

class FruitCounter:
    def __init__(self, estrategias):
        self.cfg = get_cfg()
        # Cargar tu archivo de configuración personalizado
        self.cfg.merge_from_file("/Users/miri/CV_tp/modelos/deteccion_objetos/config.yaml")
        # Cargar los pesos de tu modelo personalizado
        self.cfg.MODEL.WEIGHTS = "/Users/miri/CV_tp/modelos/deteccion_objetos/model_final.pth"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)
        self.estrategias = estrategias
        self.detected_fruits = {cls: 0 for cls in custom_classes}
        self.detected_positions = {cls: [] for cls in custom_classes}

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes
        pred_masks = instances.pred_masks
        for i, class_id in enumerate(pred_classes):
            # Usar tu metadata personalizada para obtener el nombre de la clase
            class_name = fruit_metadata.thing_classes[class_id]
            estrategia = self.estrategias.get(class_name)
            if estrategia:
                mask = pred_masks[i].numpy()
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                roi = frame_gray[y:y+h, x:x+w]
                keypoints_current, descriptors_current = orb.detectAndCompute(roi, None)
                if descriptors_current is not None:
                    estrategia.contar_frutas(mask, self.detected_fruits, self.detected_positions, frame, keypoints_current, descriptors_current)
        # Actualizar el texto de información
        info_text = ' | '.join([f"{cls.capitalize()}: {self.detected_fruits[cls]}" for cls in custom_classes])
        font_scale = 0.6 if frame.shape[0] > 400 else 0.4
        cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        return frame

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return
        self.detected_fruits = {key: 0 for key in self.detected_fruits}
        self.detected_positions = {key: [] for key in self.detected_positions}
        processed_frame = self.process_frame(image.copy())
        combined_frame = np.hstack((image, processed_frame))
        cv2.namedWindow('Original y Procesado', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original y Procesado', min(1600, combined_frame.shape[1]), min(600, combined_frame.shape[0]))
        cv2.imshow('Original y Procesado', combined_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        frame_index = 0
        cv2.namedWindow('Original y Procesado', cv2.WINDOW_NORMAL)
        self.detected_fruits = {key: 0 for key in self.detected_fruits}
        self.detected_positions = {key: [] for key in self.detected_positions}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % 25 == 0:
                frame_resized = cv2.resize(frame, (640, 360))
                processed_frame = self.process_frame(frame_resized.copy())
                combined_frame = np.hstack((frame_resized, processed_frame))
                cv2.resizeWindow('Original y Procesado', min(1600, combined_frame.shape[1]), min(600, combined_frame.shape[0]))
                cv2.imshow('Original y Procesado', combined_frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
            frame_index += 1
        cap.release()
        cv2.destroyAllWindows()

    def process_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        frame_index = 0
        cv2.namedWindow('Original y Procesado', cv2.WINDOW_NORMAL)
        self.detected_fruits = {key: 0 for key in self.detected_fruits}
        self.detected_positions = {key: [] for key in self.detected_positions}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % 32 == 0:
                frame_resized = cv2.resize(frame, (640, 360))
                processed_frame = self.process_frame(frame_resized.copy())
                combined_frame = np.hstack((frame_resized, processed_frame))
                cv2.resizeWindow('Original y Procesado', min(1600, combined_frame.shape[1]), min(600, combined_frame.shape[0]))
                cv2.imshow('Original y Procesado', combined_frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
            frame_index += 1
        cap.release()
        cv2.destroyAllWindows()

# Definir las estrategias para cada fruta
estrategias_frutas = {
    'manzana': AppleStrategy(),
    'banana': BananaStrategy(),
    'naranja': OrangeStrategy(),
    'pera': PearStrategy()
}

# Crear una instancia de FruitCounter y procesar el video
counter = FruitCounter(estrategias_frutas)
counter.process_video('mixto_1.mp4')
