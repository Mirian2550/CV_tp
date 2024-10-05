import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import supervision as sv


class FruitDetector:
    """
    Clase para detectar y contar frutas en imágenes, videos y cámara en tiempo real.
    """

    def __init__(self, model_weights, class_names, thresholds_per_class, device='cpu'):
        """
        Inicializa la clase con el modelo preentrenado, nombres de las clases y umbrales por clase.
        """
        self.class_names = class_names
        self.thresholds_per_class = thresholds_per_class

        # Configuración del modelo Detectron2
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_weights  # Cargar pesos del modelo entrenado
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)  # Ajustar el número de clases
        self.cfg.MODEL.DEVICE = device  # Usar CPU o GPU

        # Crear el predictor con la configuración del modelo
        self.predictor = DefaultPredictor(self.cfg)

        # Inicializar DeepSORT
        self.tracker = sv.ByteTrack()  # Inicializar ByteTrack solo una vez

        # Registro acumulativo de frutas detectadas
        self.total_fruit_counts = {class_name: 0 for class_name in self.class_names}
        self.counted_ids = set()

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
            for i, box in enumerate(pred_boxes):
                # Obtener la clase predicha y el puntaje
                class_id = pred_classes[i].item()
                class_name = self.class_names[class_id]
                class_threshold = self.thresholds_per_class[class_name]  # Umbral para la clase

                if scores[i] > class_threshold:  # Aplicar umbral por clase
                    # Contar la fruta detectada en este frame
                    current_frame_counts[class_name] += 1

                    # Dibujar las cajas
                    box = box.numpy()
                    start_point = (int(box[0]), int(box[1]))
                    end_point = (int(box[2]), int(box[3]))
                    frame = cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=4)

                    # Mostrar la clase y puntaje sobre la imagen, con texto más grande
                    label = f"{class_name}: {scores[i].item():.2f}"
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0,  # Tamaño aumentado
                                (0, 255, 0), 4)

        # Acumular el conteo de frutas detectadas en este frame al total
        for class_name in self.class_names:
            self.total_fruit_counts[class_name] += current_frame_counts[class_name]

        # Mostrar el conteo acumulado de frutas detectadas en la imagen, con texto más grande
        info_text = ' | '.join([f"{cls}: {count}" for cls, count in self.total_fruit_counts.items()])
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 6)  # Tamaño aumentado

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

    def process_frame_video(self, frame):
        """
        Procesa un frame de video para detectar y contar frutas, usando ByteTrack para el seguimiento.
        """
        # Realizar predicción en el frame con Detectron2
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        pred_classes = instances.pred_classes if instances.has("pred_classes") else None
        scores = instances.scores if instances.has("scores") else None

        # Preparar las detecciones para el seguimiento
        boxes = []
        confidences = []
        class_ids = []

        if pred_boxes is not None:
            for i, box in enumerate(pred_boxes):
                class_id = pred_classes[i].item()
                class_name = self.class_names[class_id]
                score = scores[i].item()
                class_threshold = self.thresholds_per_class[class_name]

                if score > class_threshold:
                    bbox = [float(box[0].item()), float(box[1].item()), float(box[2].item()), float(box[3].item())]
                    boxes.append(bbox)
                    confidences.append(score)
                    class_ids.append(class_id)

        # Solo proceder si hay detecciones
        if len(boxes) > 0:
            # Convertir las detecciones al formato esperado por ByteTrack
            detections = sv.Detections(
                xyxy=np.array(boxes),  # Usar las coordenadas en formato [x1, y1, x2, y2]
                confidence=np.array(confidences),  # Puntajes de confianza
                class_id=np.array(class_ids)  # IDs de clase
            )

            # Usar el tracker para actualizar las detecciones
            tracks = self.tracker.update_with_detections(detections)

            # Dibujar las cajas de seguimiento
            for track in tracks:
                bbox, _, score, class_id, track_id, _ = track  # Extraer los valores de la tupla
                class_name = self.class_names[class_id]

                # Dibujar la caja de seguimiento
                x1, y1, x2, y2 = map(int, bbox)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # Mostrar la clase, puntaje y el ID sobre la imagen
                label = f"{class_name} ID: {track_id} Score: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

                # Verificación de si el objeto ya ha sido contado
                if track_id not in self.counted_ids:
                    self.total_fruit_counts[class_name] += 1
                    self.counted_ids.add(track_id)  # Agregar el ID al conjunto de IDs contados

            # Ajustar el tamaño de la fuente y la posición del texto de conteo
            info_text = ' | '.join([f"{cls}: {count}" for cls, count in self.total_fruit_counts.items()])
            cv2.putText(frame, info_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0),
                        6)  # Texto 3 veces más grande

        return frame

    def process_video(self, video_path):
        """
        Procesa un video para detectar y contar frutas. Solo procesa cada 40 frames.

        Args:
            video_path (str): Ruta al archivo de video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return

        frame_count = 0
        processed_frame = None  # Inicializar para evitar errores si no se procesa ningún frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada 10 frames
            if frame_count % 10 == 0:
                processed_frame = self.process_frame_video(frame)

                # Mostrar el frame procesado
                cv2.imshow('Detección de Frutas', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir
                    break

            frame_count += 1

        cap.release()

        # Mostrar el último frame procesado hasta que se presione una tecla
        if processed_frame is not None:
            cv2.imshow('Detección de Frutas (Último Frame)', processed_frame)
            cv2.waitKey(0)  # Esperar indefinidamente hasta que se presione una tecla
            cv2.destroyAllWindows()
        else:
            print("No se procesaron frames en el video.")


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
    'banana': 0.90,
    'orange': 0.90,
    'pear': 0.05
}

# Ruta al modelo entrenado
model_weights = "./_modelos/deteccion_objetos/model_final.pth"

# Crear una instancia de FruitDetector
fruit_detector = FruitDetector(model_weights, class_names, thresholds_per_class)

# Procesar una imagen
#fruit_detector.process_image("fotos_frutas/8.jpeg")

# Procesar un video
fruit_detector.process_video("videos_frutas/2.mp4")

# Procesar el video de la cámara
# fruit_detector.process_camera()
