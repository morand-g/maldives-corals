"""This file contains the functions needed to re-train an object-detection
YOLO model using the coral images.

WIP: the current trained model has very poor performances. That's currently 
being worked on.

Note that the trained model is not provided on the repository."""

from imageai.Detection import ObjectDetection
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection


def detect_objects_untrained(input_image, output_image):
    """Run the object-detection model YOLO V3 without any further 
    training"""

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("yolov3.pt")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=input_image, output_image_path=output_image, minimum_percentage_probability=0.1)
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def detect_objects(input_image):
    """Run the object-detection model YOLO V3 after retraining
    it with the coral images"""

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("data/models/yolov3_data_last.pt")
    detector.setJsonPath("data/json/data_yolov3_detection_config.json")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=input_image, minimum_percentage_probability=0.1)
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def train_model():
    """Retrain the YOLO model with with the coral images. In order to train the model you need
    training data formatted in YOLO format. These files must be included in a data subfolder.
    The trained model will be exported to data/models.
    """
    trainer = DetectionModelTrainer()

    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="data")
    trainer.setTrainConfig(object_names_array=["acropora", "tag", "pocillopora", "dead", "bleached"], 
                           batch_size=5, 
                           num_experiments=200, 
                           train_from_pretrained_model="yolov3.pt")
    trainer.trainModel()

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from matplotlib.image import imread
from PIL import Image
import cv2

def predict_with_sam(image):
    
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    #predictor = SamPredictor(sam)
    #predictor.set_image(image)
    mask_generator = SamAutomaticMaskGenerator(sam)
    image_bgr = cv2.imread(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = mask_generator.generate(image_rgb)

predict_with_sam("image_test.jpg")





