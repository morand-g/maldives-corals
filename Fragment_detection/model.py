"""This file contains the functions needed to re-train an object-detection
YOLO model using the coral images.

WIP: the current trained model has very poor performances. That's currently 
being worked on.

Note that the trained model is not provided on the repository."""

import os
from imageai.Detection import ObjectDetection
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection
from pylabel import importer
import cv2




def coco_to_yolo(annotations_path, img_path):
    """Helper function to convert COCO annotations to YOLO format"""
    
    #Specify path to the coco.json file
    path_to_annotations = annotations_path
    #Specify the path to the images (if they are in a different folder than the annotations)
    path_to_images = img_path
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="COCO_annot")
    dataset.path_to_annotations = path_to_annotations + "/yolo"
    dataset.export.ExportToYoloV5()

def equalize_img(img_folder, output_folder):
    """Equalizes images histograms"""

    processed_img = 0
    os.makedirs(output_folder)
    files = os.listdir(img_folder)
    for img in files:
        rgb_img = cv2.imread(img_folder + '/' + img)
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(output_folder + '/' + img, equalized_img)
        processed_img += 1
        if processed_img % 10 == 0: 
            print(f"{processed_img} images processed")
    


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
    detections = detector.detectObjectsFromImage(input_image=input_image, output_image_path="prediction.jpg", minimum_percentage_probability=50)
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def train_model(model_path):
    """Retrain the YOLO model with with the coral images. In order to train the model you need
    training data formatted in YOLO format. These files must be included in a data subfolder.
    The trained model will be exported to data/models.
    """
    trainer = DetectionModelTrainer()

    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="data")
    trainer.setTrainConfig(object_names_array=["acropora", "tag", "pocillopora", "dead", "bleached"], 
                           batch_size=10, 
                           num_experiments=20, 
                           train_from_pretrained_model=model_path)
    trainer.trainModel()


# coco_to_yolo("output/COCO_train.json", "output/train_pictures")
# coco_to_yolo("output/COCO_val.json", "output/val_pictures")
# equalize_img("output/all_pictures", "eq_pictures")
# detect_objects("image_test.jpg")
train_model("yolov3_on_raw.pt")

###########################################################
# TRYING STUFF BELOW


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







