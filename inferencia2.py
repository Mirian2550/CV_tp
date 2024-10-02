import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Etiquetas de las clases (ajustar según el orden de tus clases)
class_names = ['apple', 'banana', 'orange', 'pear']

# Umbrales específicos por clase
thresholds_per_class = {
    'apple': 0.86,
    'banana': 0.90,
    'orange': 0.90,
    'pear': 0.30
}

# Configuración del modelo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./modelos/deteccion_objetos/model_final.pth"  # Ruta al modelo entrenado
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)  # Ajustar al número de clases que tienes
cfg.MODEL.DEVICE = 'cpu'  # Usar CPU

# Crear el predictor usando el modelo configurado
predictor = DefaultPredictor(cfg)

# Ruta a la imagen que quieres probar
image_path = "fotos_frutas/9.jpeg"

# Cargar la imagen
image = cv2.imread(image_path)

# Realizar la predicción usando el predictor configurado
outputs = predictor(image)

# Extraer las predicciones de la imagen
instances = outputs["instances"].to("cpu")  # Mover a CPU si está en GPU
pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
pred_classes = instances.pred_classes if instances.has("pred_classes") else None
scores = instances.scores if instances.has("scores") else None

# Dibujar las cajas de predicción sobre la imagen
if pred_boxes is not None:
    for i, box in enumerate(pred_boxes):
        # Obtener la clase predicha y el puntaje
        class_id = pred_classes[i].item()
        class_name = class_names[class_id]
        class_threshold = thresholds_per_class[class_name]  # Umbral específico para la clase

        if scores[i] > class_threshold:  # Aplicar umbral por clase
            # Dibujar las cajas
            box = box.numpy()
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            image = cv2.rectangle(image, start_point, end_point, color=(0, 255, 0), thickness=2)

            # Agregar el nombre de la clase y el puntaje
            label = f"{class_name}: {scores[i].item():.2f}"
            cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con las predicciones
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
