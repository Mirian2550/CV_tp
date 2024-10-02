import os
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

# Registrar el dataset de validación
valid_dataset_path = "valid"
register_coco_instances("fruit_dataset_valid", {}, f"{valid_dataset_path}/_annotations.coco.json", valid_dataset_path)

# Configuración del modelo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./deteccion_objetos/model_final.pth"  # Ruta al modelo entrenado
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Umbral de confianza para predicciones
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Ajustar al número de clases que tengas
cfg.DATASETS.TEST = ("fruit_dataset_valid",)  # Usar el dataset de validación
cfg.MODEL.DEVICE = 'cpu'  # Usar CPU

# Reducir número de workers para evitar problemas de multiprocessing
cfg.DATALOADER.NUM_WORKERS = 0

# Crear el predictor para cargar el modelo
predictor = DefaultPredictor(cfg)

# Evaluador para el conjunto de validación
evaluator = COCOEvaluator("fruit_dataset_valid", cfg, False, output_dir="deteccion_objetos/")
val_loader = build_detection_test_loader(cfg, "fruit_dataset_valid")

# Ejecutar la evaluación
print("Evaluando el modelo...")
metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
print(metrics)
