"""
This file contains some helper functions used by the module's models.
"""

import os
import cv2
from pylabel import importer

def coco_to_yolo(annotations_path, img_path):
    """
    Helper function to convert COCO annotations to YOLO format
    """
    
    #Specify path to the coco.json file
    path_to_annotations = annotations_path
    #Specify the path to the images (if they are in a different folder than the annotations)
    path_to_images = img_path
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="COCO_annot")
    dataset.path_to_annotations = path_to_annotations + "/yolo"
    dataset.export.ExportToYoloV5()

def equalize_img(img_folder, output_folder):
    """
    Equalizes images histograms
    """

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