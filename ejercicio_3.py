import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

class FruitStrategy(ABC):
    @abstractmethod
    def contar_frutas(self, mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current):
        pass

    def contar_y_marcar(self, fruit_name, mask, detected_fruits, detected_positions, frame, keypoints_current, descriptors_current):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

class FruitCounter:
    def __init__(self, estrategias):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)
        self.estrategias = estrategias
        self.detected_fruits = {'manzana': 0, 'banana': 0, 'naranja': 0, 'pera': 0}
        self.detected_positions = {'manzana': [], 'banana': [], 'naranja': [], 'pera': []}

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes
        pred_masks = instances.pred_masks
        for i, class_id in enumerate(pred_classes):
            class_name = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[class_id].lower()
            estrategia = self.estrategias.get(class_name)
            if estrategia:
                mask = pred_masks[i].numpy()
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                roi = frame_gray[y:y+h, x:x+w]
                keypoints_current, descriptors_current = orb.detectAndCompute(roi, None)
                if descriptors_current is not None:
                    estrategia.contar_frutas(mask, self.detected_fruits, self.detected_positions, frame, keypoints_current, descriptors_current)
        info_text = f"Manzanas: {self.detected_fruits['manzana']} | Bananas: {self.detected_fruits['banana']} | Naranjas: {self.detected_fruits['naranja']} | Peras: {self.detected_fruits['pera']}"
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

estrategias_frutas = {
    'apple': AppleStrategy(),
    'banana': BananaStrategy(),
    'orange': OrangeStrategy(),
    'pear': PearStrategy()
}

counter = FruitCounter(estrategias_frutas)
counter.process_video('mixto_1.mp4')
