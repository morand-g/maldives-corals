import os
import glob
import json
import copy
import shutil
from random import sample, random

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf

from urllib.request import urlopen
from tf_unet import image_util

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def prepare_pictures(masks_path, pictures_path, out_path, target_width = 400):

    # Resize pictures and masks, adds padding, and stores them in out_path

    i, j = 0, 0
    target_height = 0.75 * target_width

    pics_paths = glob.glob(pictures_path + "*.jpg") + glob.glob(pictures_path + "*.png")
    masks_paths = glob.glob(masks_path + "*.png")

    # Pictures
    for path in pics_paths:
        new_path = path.replace(pictures_path, out_path).replace('.jpg', '.png')

        if new_path.replace(out_path, masks_path) in masks_paths:

            image = Image.open(path)
            width, height = image.size
            new_height = int(height * target_width / width)
            size = target_width, new_height
            image.thumbnail(size, Image.ANTIALIAS)
            im = np.array(image, np.float32)

            padded = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

            cv2.imwrite(new_path, padded)
            i += 1

    # Masks
    for path in masks_paths:
        new_path = path.replace(masks_path, out_path).replace('.png', '_mask.png')
        
        image = Image.open(path)
        width, height = image.size
        new_height = int(height * target_width / width)
        size = target_width, new_height
        image = image.convert('L')
        image.thumbnail(size, Image.ANTIALIAS)
        im = np.array(image, np.float32)

        cv2.imwrite(new_path, 255*im)
        j += 1

    print("{} pictures processed\n{} masks processed".format(i, j))


def download_masks(json_path, masks_path):

    # Downloads masks from the urls specified in the json_path

    with open(json_path, "r") as f:
        data = json.load(f)

    for image in data:
        filename = image['External ID']

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


class CustomDataProvider(image_util.ImageDataProvider):

    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation

        :param data: the data array
        :param labels: the label array
        """

        # Flip

        if random() > 0.5:
            data = np.flip(data, 1)
            labels = np.flip(labels, 1)

        (h, w) = data.shape[:2]

        max_len = np.sqrt(h**2 + w**2)
        max_w, max_h = max_len, max(max_len, 0.75*w)
        vert_border = np.ceil(0.5*(max_h - h))
        hor_border = np.ceil(0.5*(max_w - w))

        # Fix ratio
        h = int(0.75 * w)

        # Add padding to get 4:3 ratio plus enough for rotation

        data_padded = cv2.copyMakeBorder(data, int(vert_border), int(vert_border), int(hor_border), int(hor_border), cv2.BORDER_REFLECT)
        labels_padded = cv2.copyMakeBorder(labels, int(vert_border), int(vert_border), int(hor_border), int(hor_border), cv2.BORDER_REFLECT)

        angle = int(random() * 360)

        (padded_h, padded_w) = data_padded.shape[:2]

        # Rotate
        
        center = (padded_w / 2, padded_h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        data_rotated = cv2.warpAffine(data_padded, M, (padded_w, padded_h))
        labels_rotated = cv2.warpAffine(labels_padded, M, (padded_w, padded_h))

        (new_h, new_w) = data_rotated.shape[:2]

        # Crop to original size plus add padding to account for cropping

        w_margins = (new_w - w) // 2, (new_w - w) // 2 + (new_w - w) % 2
        h_margins = (new_h - h) // 2, (new_h - h) // 2 + (new_h - h) % 2

        data_cropped = data_rotated[h_margins[0]:(new_h - h_margins[1]), w_margins[0]:(new_w - w_margins[1])]
        labels_cropped = labels_rotated[h_margins[0]:(new_h - h_margins[1]), w_margins[0]:(new_w - w_margins[1])]

        data_padded = cv2.copyMakeBorder(data_cropped, 200, 200, 200, 200, cv2.BORDER_REFLECT)
        labels_padded = cv2.copyMakeBorder(labels_cropped, 200, 200, 200, 200, cv2.BORDER_REFLECT)

        # cv2.imwrite('/tmp/data' + str(random()) + '.png', 255*data_padded)
        # cv2.imwrite('/tmp/labels' + str(random()) + '.png', 255*labels_padded[:,:,0])

        return cv2.cvtColor(data_padded, cv2.COLOR_BGR2RGB), labels_padded