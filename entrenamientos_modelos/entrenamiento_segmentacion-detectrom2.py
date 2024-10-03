import json
import os
import subprocess

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

# Configuración del logger
setup_logger()


class FruitSegmentation:
    """
    A class used to download, filter, modify, and train a fruit segmentation model using Detectron2.

    Attributes:
    -----------
    train_dataset_path : str
        The path to the training dataset directory.
    test_dataset_path : str
        The path to the testing dataset directory.
    output_dir : str
        The directory where the model and results will be saved.
    num_classes : int
        Number of target classes for segmentation.
    base_lr : float
        The base learning rate for the model.
    max_iter : int
        Maximum number of iterations for training.
    batch_size : int
        The batch size used for training.
    num_workers : int
        Number of worker threads used for loading data.
    device : str
        Device used for training and inference (CPU/GPU).
    classes_to_keep : list
        List of classes to keep from the dataset.
    """

    def __init__(self, train_dataset_path, test_dataset_path,
                 output_dir="segmentacion", num_classes=4, base_lr=0.0025,
                 max_iter=3000, batch_size=2, num_workers=2, device="cpu"):
        """
        Initializes the FruitSegmentation class with dataset paths, training parameters, and filtering setup.

        Parameters:
        -----------
        train_dataset_path : str
            The path to the training dataset directory.
        test_dataset_path : str
            The path to the testing dataset directory.
        output_dir : str
            The directory where the model and results will be saved.
        num_classes : int
            Number of target classes for segmentation.
        base_lr : float
            The base learning rate for the model.
        max_iter : int
            Maximum number of iterations for training.
        batch_size : int
            The batch size used for training.
        num_workers : int
            Number of worker threads used for loading data.
        device : str
            Device used for training and inference (CPU/GPU).
        """
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.classes_to_keep = ['apple', 'banana', 'orange', 'pear']

        self._download_dataset()
        self._filter_dataset()
        self.group_classes_for_segmentation(f"{self.train_dataset_path}/_annotations.coco.json")
        self.group_classes_for_segmentation(f"{self.test_dataset_path}/_annotations.coco.json")
        self._register_datasets()
        self.cfg = self._setup_cfg()

    def _download_dataset(self):
        """
        Downloads the dataset from Roboflow if it's not available locally.
        """
        if not os.path.exists(self.train_dataset_path) or not os.path.exists(self.test_dataset_path):
            print("Descargando el dataset desde Roboflow...")
            subprocess.run(
                'curl -L "https://universe.roboflow.com/ds/Z9Y94KGpRy?key=gXqyaKGyRl" > roboflow.zip',
                shell=True)
            subprocess.run("unzip roboflow.zip", shell=True)
            os.remove("../roboflow.zip")
            print("Dataset descargado, extraído y el archivo .zip eliminado.")
        else:
            print("El dataset ya está disponible.")

    def _filter_dataset(self):
        """
        Filters the dataset annotations and removes any images that do not contain the relevant classes.
        """

        def filter_annotations(json_path, dataset_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            filtered_images = []
            filtered_annotations = []
            class_ids_to_keep = [
                data['categories'].index(c) for c in data['categories'] if c['name'] in self.classes_to_keep
            ]

            for annotation in data['annotations']:
                if annotation['category_id'] in class_ids_to_keep:
                    filtered_annotations.append(annotation)

            for image in data['images']:
                image_annotations = [a for a in filtered_annotations if a['image_id'] == image['id']]
                if image_annotations:
                    filtered_images.append(image)
                else:
                    image_path = os.path.join(dataset_path, image['file_name'])
                    if os.path.exists(image_path):
                        os.remove(image_path)

            data['images'] = filtered_images
            data['annotations'] = filtered_annotations
            data['categories'] = [c for c in data['categories'] if c['name'] in self.classes_to_keep]

            with open(json_path, 'w') as f:
                json.dump(data, f)

        filter_annotations(f"{self.train_dataset_path}/_annotations.coco.json", self.train_dataset_path)
        filter_annotations(f"{self.test_dataset_path}/_annotations.coco.json", self.test_dataset_path)

    def group_classes_for_segmentation(self, json_path):
        """
        Groups the classes into new categories and updates the segmentation annotations, overwriting the original file.

        Parameters:
        -----------
        json_path : str
            Path to the COCO format annotation file to be overwritten.
        """

        class_mapping = {
            1: 'apple', 14: 'apple', 16: 'apple',  # Manzanas
            3: 'banana',  # Bananas
            30: 'orange',  # Naranjas
            12: 'pear', 32: 'pear', 55: 'pear'  # Peras
        }

        new_categories = [
            {"id": 1, "name": "apple"},
            {"id": 2, "name": "banana"},
            {"id": 3, "name": "orange"},
            {"id": 4, "name": "pear"}
        ]

        new_category_map = {"apple": 1, "banana": 2, "orange": 3, "pear": 4}

        with open(json_path, 'r') as f:
            data = json.load(f)

        filtered_annotations = []
        filtered_images = []

        for annotation in data['annotations']:
            original_class_id = annotation['category_id']
            if original_class_id in class_mapping:
                new_class_name = class_mapping[original_class_id]
                annotation['category_id'] = new_category_map[new_class_name]
                filtered_annotations.append(annotation)

        for image in data['images']:
            image_annotations = [a for a in filtered_annotations if a['image_id'] == image['id']]
            if image_annotations:
                filtered_images.append(image)

        data['annotations'] = filtered_annotations
        data['images'] = filtered_images
        data['categories'] = new_categories

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Archivo JSON sobrescrito con nuevas clases agrupadas: {json_path}")

    def _register_datasets(self):
        """
        Registers the filtered datasets for training and testing in the Detectron2 framework.
        """
        register_coco_instances("fruit_dataset_train", {},
                                f"{self.train_dataset_path}/_annotations.coco.json", self.train_dataset_path)
        register_coco_instances("fruit_dataset_test", {},
                                f"{self.test_dataset_path}/_annotations.coco.json", self.test_dataset_path)

    def _setup_cfg(self):
        """
        Sets up the configuration for the Detectron2 segmentation model and training parameters.
        """
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("fruit_dataset_train",)
        cfg.DATASETS.TEST = ("fruit_dataset_test",)
        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.SOLVER.BASE_LR = self.base_lr
        cfg.SOLVER.MAX_ITER = self.max_iter
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.OUTPUT_DIR = self.output_dir
        cfg.MODEL.DEVICE = self.device
        os.makedirs(self.output_dir, exist_ok=True)
        return cfg

    def train(self, resume=False):
        """
        Starts the training process for the segmentation model.

        Parameters:
        -----------
        resume : bool
            Whether to resume training from the last checkpoint.
        """
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=resume)
        trainer.train()


if __name__ == "__main__":
    train_dataset_path = "train"
    test_dataset_path = "test"

    fruit_segmentor = FruitSegmentation(train_dataset_path, test_dataset_path)
    fruit_segmentor.train(resume=False)
