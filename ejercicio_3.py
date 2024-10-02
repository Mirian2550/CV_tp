import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from scipy.spatial import distance as dist
import numpy as np


class CentroidTracker:
    """
    Rastreador basado en el centrado de las frutas detectadas para asignar un ID único a cada detección.
    """

    def __init__(self, max_disappeared=50):
        """
        Inicializa el rastreador de centroides.

        Args:
            max_disappeared (int): Número máximo de frames en los que un objeto puede desaparecer antes de ser removido.
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """
        Registra un nuevo objeto con un ID único.

        Args:
            centroid (tuple): Coordenadas del centroide del objeto detectado.
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Remueve el objeto del seguimiento.

        Args:
            object_id (int): ID del objeto a remover.
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Actualiza las posiciones de los objetos detectados y asigna IDs a los nuevos objetos.

        Args:
            rects (list): Lista de cajas delimitadoras (x1, y1, x2, y2) de los objetos detectados.

        Returns:
            dict: Diccionario con los IDs de los objetos y sus coordenadas.
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


class FruitDetector:
    """
    Clase para detectar, contar y hacer tracking de frutas en imágenes, videos y cámara en tiempo real.
    """

    def __init__(self, model_weights, class_names, thresholds_per_class, device='cpu'):
        """
        Inicializa la clase con el modelo preentrenado, nombres de las clases y umbrales por clase.

        Args:
            model_weights (str): Ruta a los pesos del modelo entrenado.
            class_names (list): Lista de nombres de las clases.
            thresholds_per_class (dict): Diccionario con los umbrales de confianza para cada clase.
            device (str): Dispositivo a utilizar ('cpu' o 'cuda').
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

        # Registro acumulativo de frutas detectadas
        self.total_fruit_counts = {class_name: 0 for class_name in self.class_names}

        # Inicializar el rastreador de centroides
        self.tracker = CentroidTracker()

    def process_frame(self, frame):
        """
        Procesa un frame para detectar, contar y rastrear frutas.

        Args:
            frame (numpy.array): Imagen o frame a procesar.

        Returns:
            frame (numpy.array): Frame procesado con cajas, IDs y conteo de frutas.
        """
        # Realizar predicción en el frame
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        pred_classes = instances.pred_classes if instances.has("pred_classes") else None
        scores = instances.scores if instances.has("scores") else None

        rects = []
        current_frame_counts = {class_name: 0 for class_name in self.class_names}

        if pred_boxes is not None:
            for i, box in enumerate(pred_boxes):
                class_id = pred_classes[i].item()
                class_name = self.class_names[class_id]
                class_threshold = self.thresholds_per_class[class_name]

                if scores[i] > class_threshold:
                    box_np = box.numpy()
                    rects.append(box_np)

                    # Dibujar las cajas
                    start_point = (int(box_np[0]), int(box_np[1]))
                    end_point = (int(box_np[2]), int(box_np[3]))
                    cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

                    # Acumular conteo
                    current_frame_counts[class_name] += 1

        # Actualizar el rastreo de centroides
        objects = self.tracker.update(rects)

        # Mostrar los IDs y las coordenadas de los objetos rastreados
        for object_id, centroid in objects.items():
            text = f"ID {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Mostrar el conteo acumulado de frutas detectadas en la imagen
        for class_name in self.class_names:
            self.total_fruit_counts[class_name] += current_frame_counts[class_name]

        return frame, self.total_fruit_counts

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

        processed_frame, fruit_counts = self.process_frame(image)

        # Mostrar la imagen procesada
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    def process_video(self, video_path):
        """
        Procesa un video para detectar y rastrear frutas.
        Args:
            video_path (str): Ruta al archivo de video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return

        print("Procesando video...")  # Verificación de que el bucle ha comenzado

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer el frame.")
                break

            # Detectar y rastrear las frutas
            processed_frame, fruit_counts = self.process_frame(frame)

            # Mostrar el frame procesado en una ventana
            cv2.imshow('Detección y Rastreo de Frutas', processed_frame)

            # Mostrar el conteo acumulado en otra ventana
            count_frame = np.zeros((200, 400, 3), dtype="uint8")
            info_text = ' | '.join([f"{cls}: {count}" for cls, count in fruit_counts.items()])
            cv2.putText(count_frame, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('Conteo de Frutas', count_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir
                break

        cap.release()
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
                processed_frame, fruit_counts = self.process_frame(frame)

                # Mostrar el frame procesado
                cv2.imshow('Detección de Frutas (Cámara)', processed_frame)

                # Mostrar el conteo acumulado en otra ventana
                count_frame = np.zeros((200, 400, 3), dtype="uint8")
                info_text = ' | '.join([f"{cls}: {count}" for cls, count in fruit_counts.items()])
                cv2.putText(count_frame, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow('Conteo de Frutas', count_frame)

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
    'pear': 0.30
}

# Ruta al modelo entrenado
model_weights = "./modelos/deteccion_objetos/model_final.pth"

# Crear una instancia de FruitDetector
fruit_detector = FruitDetector(model_weights, class_names, thresholds_per_class)

# Procesar una imagen
# fruit_detector.process_image("fotos_frutas/8.jpeg")

# Procesar un video
fruit_detector.process_video("videos_frutas/1.mp4")

# Procesar el video de la cámara
# fruit_detector.process_camera()
