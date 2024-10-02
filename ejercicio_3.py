import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np


def calculate_iou(box1, box2):
    """
    Calcula el IOU (Intersection over Union) entre dos cajas.

    Args:
        box1: Primera caja [x1, y1, x2, y2]
        box2: Segunda caja [x1, y1, x2, y2]

    Returns:
        iou (float): Valor de intersección sobre unión.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou


class FruitDetector:
    """
    Clase para detectar y contar frutas en imágenes, videos y cámara en tiempo real.
    """

    def __init__(self, model_weights, class_names, thresholds_per_class, device='cpu', iou_threshold=0.4):
        """
        Inicializa la clase con el modelo preentrenado, nombres de las clases y umbrales por clase.

        Args:
            model_weights (str): Ruta a los pesos del modelo entrenado.
            class_names (list): Lista de nombres de las clases.
            thresholds_per_class (dict): Diccionario con los umbrales de confianza para cada clase.
            device (str): Dispositivo a utilizar ('cpu' o 'cuda').
            iou_threshold (float): Umbral de IOU para considerar dos cajas como la misma detección.
        """
        self.class_names = class_names
        self.thresholds_per_class = thresholds_per_class
        self.iou_threshold = iou_threshold

        # Configuración del modelo Detectron2
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_weights  # Cargar pesos del modelo entrenado
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)  # Ajustar el número de clases
        self.cfg.MODEL.DEVICE = device  # Usar CPU o GPU

        # Crear el predictor con la configuración del modelo
        self.predictor = DefaultPredictor(self.cfg)

        # Registro acumulativo de frutas detectadas
        self.total_fruit_counts = {class_name: 0 for class_name in self.class_names}

        # Mantener un historial de las posiciones de las frutas detectadas para evitar duplicados
        self.previous_boxes = []  # Para guardar las cajas de detecciones previas

    def process_frame(self, frame):
        """
        Procesa un frame para detectar y contar frutas.

        Args:
            frame (numpy.array): Imagen o frame a procesar.

        Returns:
            frame (numpy.array): Frame procesado con cajas y conteo de frutas.
        """
        # Realizar predicción en el frame
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        pred_classes = instances.pred_classes if instances.has("pred_classes") else None
        scores = instances.scores if instances.has("scores") else None

        # Contador de frutas detectadas en este frame
        current_frame_counts = {class_name: 0 for class_name in self.class_names}

        # Dibujar las cajas de predicción sobre el frame y contar frutas
        if pred_boxes is not None:
            new_boxes = []  # Guardar las cajas del frame actual

            for i, box in enumerate(pred_boxes):
                # Obtener la clase predicha y el puntaje
                class_id = pred_classes[i].item()
                class_name = self.class_names[class_id]
                class_threshold = self.thresholds_per_class[class_name]  # Umbral para la clase

                if scores[i] > class_threshold:  # Aplicar umbral por clase
                    box_np = box.numpy()
                    is_duplicate = False

                    # Verificar si la caja detectada se superpone con alguna caja anterior (IOU)
                    for prev_box in self.previous_boxes:
                        iou = calculate_iou(box_np, prev_box)
                        if iou > self.iou_threshold:  # Consideramos la fruta ya detectada si el IOU es mayor al umbral
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        # Si no es un duplicado, contar la fruta
                        current_frame_counts[class_name] += 1
                        new_boxes.append(box_np)

                        # Dibujar las cajas
                        start_point = (int(box_np[0]), int(box_np[1]))
                        end_point = (int(box_np[2]), int(box_np[3]))
                        frame = cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=4)

                        # Mostrar la clase y puntaje sobre la imagen, con texto más grande
                        label = f"{class_name}: {scores[i].item():.2f}"
                        cv2.putText(frame, label, (int(box_np[0]), int(box_np[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                                    (0, 255, 0), 4)

            # Actualizar las cajas anteriores con las del frame actual
            self.previous_boxes = new_boxes

        # Acumular el conteo de frutas detectadas en este frame al total
        for class_name in self.class_names:
            self.total_fruit_counts[class_name] += current_frame_counts[class_name]

        # Mostrar el conteo acumulado de frutas detectadas en la imagen, con texto más grande
        info_text = ' | '.join([f"{cls}: {count}" for cls, count in self.total_fruit_counts.items()])
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 6)

        return frame

    def process_image(self, image_path):
        """
        Procesa una imagen para detectar frutas y muestra el resultado.

        Args:
            image_path (str): Ruta a la imagen que se va a procesar.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen en {image_path}")
            return

        processed_frame = self.process_frame(image)

        # Mostrar la imagen procesada
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    def process_video(self, video_path):
        """
        Procesa un video para detectar frutas. Solo procesa cada 40 frames.

        Args:
            video_path (str): Ruta al archivo de video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada 40 frames
            if frame_count % 40 == 0:
                processed_frame = self.process_frame(frame)

                # Mostrar el frame procesado
                cv2.imshow('Detección de Frutas', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir
                    break

            frame_count += 1

        cap.release()

        # Mostrar el último frame hasta que se presione una tecla
        cv2.imshow('Detección de Frutas (Último Frame)', processed_frame)
        cv2.waitKey(0)  # Esperar indefinidamente hasta que se presione una tecla
        cv2.destroyAllWindows()

    def process_camera(self):
        """
        Procesa el video capturado desde la cámara para detectar frutas. Solo procesa cada 40 frames.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara")
            return

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada 40 frames
            if frame_count % 40 == 0:
                processed_frame = self.process_frame(frame)

                # Mostrar el frame procesado
                cv2.imshow('Detección de Frutas (Cámara)', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir
                    break

            frame_count += 1

        cap.release()

        # Mostrar el último frame hasta que se presione una tecla
        cv2.imshow('Detección de Frutas (Último Frame)', processed_frame)
        cv2.waitKey(0)  # Esperar indefinidamente hasta que se presione una tecla
        cv2.destroyAllWindows()


# Configuración de las clases y umbrales
class_names = ['apple', 'banana', 'orange', 'pear']
thresholds_per_class = {
    'apple': 0.86,
    'banana': 0.80,
    'orange': 0.90,
    'pear': 0.30
}

# Ruta al modelo entrenado
model_weights = "./modelos/deteccion_objetos/model_final.pth"

# Crear una instancia de FruitDetector
fruit_detector = FruitDetector(model_weights, class_names, thresholds_per_class)

# Procesar una imagen
#fruit_detector.process_image("fotos_frutas/7.jpeg")

# Procesar un video
fruit_detector.process_video("videos_frutas/1.mp4")

# Procesar el video de la cámara
#fruit_detector.process_camera()
