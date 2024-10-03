import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np

# Etiquetas de las clases (ajustar según el orden de tus clases)
class_names = ['apple', 'banana', 'orange', 'pear']

# Umbrales específicos por clase
thresholds_per_class = {
    'apple': 0.85,
    'banana': 0.85,
    'orange': 0.85,
    'pear': 0.15
}

# Configuración del modelo de segmentación
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./modelos/segmentacion/model_final.pth"  # Ruta al modelo entrenado
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)  # Ajustar al número de clases
cfg.MODEL.DEVICE = 'cpu'  # Usar CPU

# Crear el predictor usando el modelo configurado
predictor = DefaultPredictor(cfg)

# Ruta a la imagen que quieres probar
image_path = "fotos_frutas/8.jpeg"

# Cargar la imagen
image = cv2.imread(image_path)

# Realizar la predicción usando el predictor configurado
outputs = predictor(image)

# Extraer las predicciones de la imagen (clases, puntuaciones, máscaras)
instances = outputs["instances"].to("cpu")  # Mover a CPU si está en GPU
pred_classes = instances.pred_classes if instances.has("pred_classes") else None
scores = instances.scores if instances.has("scores") else None
pred_masks = instances.pred_masks if instances.has("pred_masks") else None  # Obtener las máscaras

# Dibujar las máscaras sobre la imagen
if pred_masks is not None:
    mask_image = np.zeros_like(image)

    for i, mask in enumerate(pred_masks):
        # Obtener la clase predicha y el puntaje
        class_id = pred_classes[i].item()
        class_name = class_names[class_id]
        class_threshold = thresholds_per_class[class_name]  # Umbral específico para la clase

        if scores[i] > class_threshold:  # Aplicar umbral por clase
            # Dibujar las máscaras sobre la imagen
            mask = mask.numpy()
            mask_image[mask] = [0, 255, 0]  # Máscara verde para la clase detectada

            # Agregar el nombre de la clase y el puntaje
            label = f"{class_name}: {scores[i].item():.2f}"
            cv2.putText(image, label, (mask.nonzero()[1][0], mask.nonzero()[0][0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

# Combinar las máscaras con la imagen original
combined_image = cv2.addWeighted(image, 1, mask_image, 0.5, 0)

# Mostrar la imagen con las predicciones y máscaras
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
