import os
import glob
import json
import copy
import shutil
from random import sample

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf

from urllib.request import urlopen
from numpy import random

class CustomDataProvider:
    
    def __init__(self, data_folder, mask_folder, **kwargs):
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.kwargs = kwargs
    
    def load_data(self, id):
        data_file = self.data_folder + '/' + id + '.png'
        mask_file = self.mask_folder + '/' + id + '.png'
        
        data = np.array(Image.open(data_file))
        mask = np.array(Image.open(mask_file))
        
        return data, mask
    
    def _post_process(self, data, labels):
        # Flip
        if random() > 0.5:
            data = np.fliplr(data)
            labels = np.fliplr(labels)
        
        # Fix ratio
        (h, w) = data.shape[:2]
        new_h = int(0.75 * w)
        data = np.pad(data, ((0, new_h - h), (0, 0), (0, 0)), mode='reflect')
        labels = np.pad(labels, ((0, new_h - h), (0, 0), (0, 0)), mode='reflect')
        
        # Rotate
        angle = int(random() * 360)
        data = Image.fromarray(data)
        labels = Image.fromarray(labels)
        data = data.rotate(angle, resample=Image.BICUBIC, fillcolor='reflect')
        labels = labels.rotate(angle, resample=Image.BICUBIC, fillcolor='reflect')
        data = np.array(data)
        labels = np.array(labels)
        
        # Crop to original size plus add padding to account for cropping
        (new_h, new_w) = data.shape[:2]
        w_margins = (new_w - w) // 2, (new_w - w) // 2 + (new_w - w) % 2
        h_margins = (new_h - h) // 2, (new_h - h) // 2 + (new_h - h) % 2
        data = data[h_margins[0]:(new_h - h_margins[1]), w_margins[0]:(new_w - w_margins[1])]
        labels = labels[h_margins[0]:(new_h - h_margins[1]), w_margins[0]:(new_w - w_margins[1])]
        data = np.pad(data, ((200, 200), (200, 200), (0, 0)), mode='reflect')
        labels = np.pad(labels, ((200, 200), (200, 200), (0, 0)), mode='reflect')
        
        # Return processed data and labels
        return data, labels

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def prepare_pictures(masks_path, pictures_path, out_path, target_width=400):

    # Redimensionne les images et les masques, ajoute un remplissage et les stocke dans out_path
    
    # Initialiser les compteurs
    i, j = 0, 0
    
    # Définir la hauteur cible comme 0,75 de la largeur cible
    target_height = 0.75 * target_width

    # Extraire les chemins de tous les fichiers d'image avec les extensions .jpg et .png dans pictures_path
    pics_paths = glob.glob(os.path.join(pictures_path, "*.jpg")) + glob.glob(os.path.join(pictures_path, "*.png"))
    

    # Extraire les chemins de tous les fichiers de masque avec l'extension .png dans masks_path
    masks_paths = glob.glob(os.path.join(masks_path, "*.png"))
    
    # Traiter les images
    for path in pics_paths:
        # Construire un nouveau chemin de sortie pour l'image
        new_path = os.path.join(out_path, os.path.basename(path).replace('.jpg', '.png'))

        # Vérifier si le chemin de sortie pour le masque existe
        if os.path.join(out_path, os.path.basename(path).replace('.jpg', '_mask.png')):

            # Ouvrir l'image
            image = Image.open(path)
            
            # Obtenir la largeur et la hauteur de l'image d'origine
            width, height = image.size
            
            # Calculer la hauteur cible en multipliant la largeur cible par 0.75 (pour maintenir le format de l'image)
            new_height = int(height * target_width / width)
            
            # Redimensionner l'image en utilisant la fonction thumbnail() avec l'interpolation ANTIALIAS pour une meilleure qualité
            size = target_width, new_height
            image.thumbnail(size, Image.ANTIALIAS)
            
            # Convertir l'image en un tableau numpy de type flottant (float32)
            im = np.array(image, np.float32)

            # Ajouter un remplissage à l'image pour que sa dimension soit identique à celle de l'image cible
            padded = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

            # Enregistrer l'image sous forme de fichier PNG en utilisant la fonction cv2.imwrite()
            cv2.imwrite(new_path, padded)
            
            # Incrémenter le compteur pour les images traitées
            i += 1

    # Traiter les masques
    for path in masks_paths:
        
        # Construire un nouveau chemin de sortie pour le masque
        new_path = os.path.join(out_path, os.path.basename(path)[:-4] + '_mask.png')
        
        # Ouvrir le masque
        image = Image.open(path)
        
        # Obtenir la largeur et la hauteur du masque d'origine
        width, height = image.size
        
        # Calculer la hauteur cible en multipliant la largeur cible par 0.75 (pour maintenir le format de l'image)
        new_height = int(height * target_width / width)
        
        # Redimensionner le masque en utilisant la fonction thumbnail() avec l'interpolation ANTIALIAS pour une meilleure qualité
        size = target_width, new_height
        image = image.convert('L')
        image.thumbnail(size, Image.ANTIALIAS)
        
        # Convertir le masque en un tableau numpy de type flottant (float32)
        im = np.array(image, np.float32)

        # Enregistrer le masque sous forme de fichier PNG en utilisant la fonction cv2.imwrite() en multipliant chaque pixel par 255 pour le convertir en noir et blanc
        cv2.imwrite(new_path, 255*im)
        
        # Incrémenter le compteur pour les masques traités
        j += 1

    # Afficher le nombre d'images et de masques traités
    print("{} images traitées\n{} masques traités".format(i, j))


def download_masks(json_path, masks_path):

    # Downloads masks from the urls specified in the json_path

    with open(json_path) as f:
        data = json.load(f)

    for image in data:
        filename = image['External ID']
        print(filename)

        if(not(os.path.exists(masks_path + filename[:-4] + '.png'))):
            url = image['Label']['objects'][0]['instanceURI']
            resp = urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, 0)
            image = np.ceil(image / 256)
            cv2.imwrite(masks_path + filename[:-4] + '.png', image)


def check_images(path):

    # Checks picture and mask inside TFRecord

    reconstructed_images = []
    record_iterator = tf.python_io.tf_record_iterator(path=path)

    for string_record in record_iterator:
        
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['image/height'].int64_list.value[0])
        width = int(example.features.feature['image/width'].int64_list.value[0])
        img_string = (example.features.feature['image/encoded'].bytes_list.value[0])
        annotation_string = (example.features.feature['image/object/mask'].bytes_list.value[0])
        
        img = tf.image.decode_jpeg(img_string)
        reconstructed_img = tf.reshape(img, [height, width, 3])
        
        annot = tf.image.decode_png(annotation_string)
        reconstructed_annot = tf.reshape(annot, [height, width])

        sess = tf.Session()
        with sess.as_default():
            cv2.imwrite("pic.jpg", reconstructed_img.eval())
            cv2.imwrite("annot.png", reconstructed_annot.eval())

"""
def create_coco(pictures_path, data_dir, masks_path):

    # Creates COCO annotations for Tensorflow mask RCNN

    trainDict = dict()
    trainDict["categories"] = [{"supercategory": "frame", "id": 1, "name": "frame"}]
    valDict = copy.deepcopy(trainDict)

    a_count = 0
    i_count = 0

    train_images = list()
    val_images = list()

    train_annotations = list()
    val_annotations = list()

    picture_subs = {True: data_dir+"train_pictures/",
                False: data_dir+"val_pictures/"}

    shutil.rmtree(picture_subs[True])
    shutil.rmtree(picture_subs[False])
    os.mkdir(picture_subs[True])
    os.mkdir(picture_subs[False])

    image_names = []
    for path in glob.glob(pictures_path + "*.jpg"):  
        image_names.append(os.path.basename(path))

    train = sample(range(0,len(image_names)),int(len(image_names)*0.8))

    annots={True:train_annotations,False:val_annotations}
    imgs={True:train_images,False:val_images}

    for file in image_names:

        image = dict()
        image['file_name'] = file
        im = cv2.imread(pictures_path + file)
        image['height'], image['width'], _ = im.shape

        image['id'] = i_count
        imgs[i_count in train].append(image)

        cv2.imwrite(picture_subs[i_count in train] + file, im)
        
        if (os.path.exists(masks_path + file[:-4] + '.png')):
	        mask = cv2.imread(masks_path + file[:-4] + '.png', 0)
            nonzeros = np.asarray(np.nonzero(mask))
            xmin, xmax = min(nonzeros[1]), max(nonzeros[1])
	        ymin, ymax = min(nonzeros[0]), max(nonzeros[0])

	        id1 = 0
	        annotation = dict()

	        a_count = a_count + 1
	        annotation["iscrowd"] = 0
	        annotation["image_id"] = i_count
	        x1 = int(xmin)
	        y1 = int(ymin)
	        width = int(xmax) - x1
	        height = int(ymax) - y1
	        annotation["bbox"] = [x1, y1, width, height]
	        annotation["area"] = float(width * height)
	        annotation["category_id"] = 1
	        annotation["ignore"] = 0
	        annotation["id"] = id1
	        annotation["mask"] = masks_path + file[:-4] + '.png'                        

	        annots[i_count in train].append(annotation)
	        id1 += 1
	        i_count += 1

    print("Create COCO : Converted {} annotations from {} images".format(a_count,i_count))

    trainDict["images"] = imgs[True]
    trainDict["annotations"] = annots[True]
    trainDict["type"] = "instances"
    jsonString = json.dumps(trainDict, indent = 4)
    with open(data_dir + "COCO_train.json", "w") as f:
        f.write(jsonString)

    valDict["images"] = imgs[False]
    valDict["annotations"] = annots[False]
    valDict["type"] = "instances"
    jsonString = json.dumps(valDict, indent = 4)
    with open(data_dir + "COCO_val.json", "w") as f:
        f.write(jsonString)
"""


class CustomDataProvider:
    
    def __init__(self, data_folder, data_suffix, mask_suffix):
        self.data_folder = data_folder
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
    
    def load_data(self, id):
        data_file = self.data_folder + '/' + id + self.data_suffix
        mask_file = self.data_folder + '/' + id + self.mask_suffix
        
        data = np.array(Image.open(data_file))
        mask = np.array(Image.open(mask_file))
        
        return data, mask
    
    def _post_process(self, data, labels):
        # Flip
        if random() > 0.5:
            data = np.fliplr(data)
            labels = np.fliplr(labels)
        
        # Fix ratio
        (h, w) = data.shape[:2]
        new_h = int(0.75 * w)
        data = np.pad(data, ((0, new_h - h), (0, 0), (0, 0)), mode='reflect')
        labels = np.pad(labels, ((0, new_h - h), (0, 0), (0, 0)), mode='reflect')
        
        # Rotate
        angle = int(random() * 360)
        data = Image.fromarray(data)
        labels = Image.fromarray(labels)
        data = data.rotate(angle, resample=Image.BICUBIC, fillcolor='reflect')
        labels = labels.rotate(angle, resample=Image.BICUBIC, fillcolor='reflect')
        data = np.array(data)
        labels = np.array(labels)
        
        # Crop to original size plus add padding to account for cropping
        (new_h, new_w) = data.shape[:2]
        w_margins = (new_w - w) // 2, (new_w - w) // 2 + (new_w - w) % 2
        h_margins = (new_h - h) // 2, (new_h - h) // 2 + (new_h - h) % 2
        data = data[h_margins[0]:(new_h - h_margins[1]), w_margins[0]:(new_w - w_margins[1])]
        labels = labels[h_margins[0]:(new_h - h_margins[1]), w_margins[0]:(new_w - w_margins[1])]
        data = np.pad(data, ((200, 200), (200, 200), (0, 0)), mode='reflect')
        labels = np.pad(labels, ((200, 200), (200, 200), (0, 0)), mode='reflect')
        
        # Return processed data and labels
        return data, labels
