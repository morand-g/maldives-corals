import numpy as np

import cv2
from PIL import Image, ImageOps, ImageFile
import json
from sqlfunctions import insert_annotations

import os
import glob
from mpi4py import MPI

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tf_unet import unet

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
ImageFile.LOAD_TRUNCATED_IMAGES = True

vis_util.STANDARD_COLORS = ['Black', 'Yellow', 'Coral', 'Crimson', 'White']

#########################################################################################
#                                                                                       #
# This file contains all the needed functions to use previously trained neural networks #
#                 A description is included in each function definition                 #
#                                                                                       #
#########################################################################################

def export_to_json(image_folder, image_name, boxes, classes, scores, category_index, threshold = 0.8):
    
    # Exports annotations to Sloth JSON for manual correction

    classes_codes = {'Pocillopora':'poc', 'Acropora':'acro', 'Dead Coral':'dead', 'Bleached Coral':'bleached', 'Frame Tag':'tag'}
    data = {}
    data['annotations'] = []
    
    try:
        img = Image.open(image_folder + image_name)
        w, h = img.size
        
        for i in range(boxes.shape[0]):
            if(scores[i] >= threshold):
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                Type = classes_codes[str(category_index[classes[i]]['name'])]
                TopLeftX = xmin*w
                TopLeftY = ymin*h
                Width = (xmax - xmin)*w
                Height = (ymax - ymin)*h
                
                data['annotations'].append({
                    'class': Type,
                    'height': Height,
                    'width': Width,
                    'x': TopLeftX,
                    'y': TopLeftY
                })
            
        data['class'] = 'image'
        data['filename'] = image_name
        return(json.dumps(data, indent = 4))
        
    except FileNotFoundError:
        print('File {} doesn\'t exist in path {}'.format(image_name,PATH_TO_TEST_IMAGES_DIR_Low))


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):

    # Returns a dictionary of the objects that were detected on the given picture

    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict


def load_graph(model):

    # Loads the inference graph into memory

    PATH_TO_FROZEN_GRAPH = '/media/mdc/Storage/exported_model/' + model + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = '/media/mdc/Storage/data/Fragment detection/TFRecords/label_map.pbtxt'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    return(detection_graph, category_index)


def prepare_image(image_name, image_folder, resize = True):

    # Apply Autocontrast and CLAHE to picture
    im = Image.open(image_folder + image_name)
    r, g, b = im.split()
    r, g, b = ImageOps.autocontrast(r, cutoff = 1), ImageOps.autocontrast(g, cutoff = 1), ImageOps.autocontrast(b, cutoff = 1)
    im = Image.merge("RGB",[r, g, b])

    img_file = np.uint8(im)
    height, width = img_file.shape[:2]

    image_lab = cv2.cvtColor(img_file, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged_channels = cv2.merge((cl, a_channel, b_channel))
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)

    if resize and width > 1440:
        ratio = height/width
        size = (1440, int(1440 * ratio))
        final_image = cv2.resize(final_image, size, interpolation=cv2.INTER_AREA)

    return(final_image)


def save_annotations(image_name, root_folder, output_dict, category_index, insert = False, export = False, save_pictures = False, threshold = 0.8):

    # Saves annotations, according to given options :
    # - Insert into SQL database?
    # - Export JSON file? (Sloth format for correction)
    # - Save pictures with bounding boxes?

    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']

    image_folder = root_folder + "Originals/"
    json_folder = root_folder + "JSON/"
    annotations_folder = root_folder + "Annotations/"

    if(insert):
        insert_annotations(image_name, boxes, classes, scores, category_index, threshold = threshold)
    if(export):
        json_string = export_to_json(image_folder, image_name, boxes, classes, scores, category_index, threshold = threshold)

        output_file = json_folder + image_name[:-4] + '.json'
        with open(output_file, "w") as text_file:
            print(f"{json_string}", file=text_file)

    if(save_pictures):
        image_np = prepare_image(image_name, image_folder, resize = False)
        pic = vis_util.visualize_boxes_and_labels_on_image_array(image_np, boxes, classes, scores, category_index,
            use_normalized_coordinates=True, line_thickness=int(8 * image_np.shape[1] / 1440), min_score_thresh = threshold)

        cv2.imwrite(annotations_folder + image_name, cv2.cvtColor(pic,cv2.COLOR_BGR2RGB))


def mergeJSON(root_folder):

    # Merges JSON files from multiple pictures to use for batch correction in Sloth

    result = []
    for f in glob.glob(root_folder + "JSON/*.json"):
        with open(f, "r") as infile:
            d = json.load(infile)
            if(len(d["annotations"]) >= 1):
                result.append(d)

    with open(root_folder + "JSON/all_annotations.json", "w") as outfile:
        json.dump(result, outfile, indent = 4)


def load_pictures_unet(in_path, target_width = 400):

    # Prepare pictures for inference

    images = []
    metadata = []

    for path in glob.glob(in_path + "*.jpg") + glob.glob(in_path + "*.JPG"):
        
        im_name = os.path.basename(path)

        image = Image.open(path)
        width, height = image.size
        new_height = int(height * target_width / width)
        size = target_width, new_height
        resize_scale = target_width / width
        image.thumbnail(size, Image.ANTIALIAS)
                   
        im = np.array(image, np.float32)

        h_diff = int(0.75 * target_width - new_height)
        vert_border = h_diff // 2, h_diff // 2 + h_diff % 2

        padded = cv2.copyMakeBorder(im, 200 + vert_border[0], 200 + vert_border[1], 200, 200, cv2.BORDER_REFLECT)
        padded = cv2.cvtColor(padded,cv2.COLOR_BGR2RGB)

        data = np.fabs(padded)
        data -= np.amin(data)

        if np.amax(data) != 0:
            data /= np.amax(data)

        images.append(data)
        metadata.append([im_name, resize_scale])

    return(np.array(images), metadata)


def get_unet_masks(model, images, root_folder, metadata, target_width = 400, chunk_size = 10):

    # applies unet model to picture and returns frame mask

    margin = 0.05 # Remove bars detected close to the edges
    target_height = int(0.75 * target_width)

    model_path = '/media/mdc/Storage/exported_model/' + model + '/model.ckpt'
    predictions = []

    chunks = [np.array(images[x:(x + chunk_size)]) for x in range(0, len(images), chunk_size)]

    net = unet.Unet(layers=6, features_root=64, channels=3, n_class=2)

    for chunk in chunks:
        predictions.extend(net.predict(model_path, chunk, log = False))

    preds = np.array(predictions)
    pics = (preds[...,1] >= 0.1)

    masks = []

    for i in range(len(images)):
        mask = pics[i]
        m_h, m_w = mask.shape[:2]

        # Fix size

        h_diff, w_diff = target_height - m_h, target_width - m_w

        if h_diff > 0:
            padding = h_diff // 2, h_diff // 2 + h_diff % 2
            mask = np.pad(mask, ((padding[0], padding[1]), (0, 0)), mode = 'constant')
        else:
            cropping = (-h_diff) // 2, (-h_diff) // 2 + (-h_diff) % 2
            mask = mask[cropping[0]:(m_h - cropping[1])]

        if w_diff > 0:
            padding = w_diff // 2, w_diff // 2 + w_diff % 2
            mask = np.pad(mask, ((0, 0), (padding[0], padding[1])), mode = 'constant')
        else:
            cropping = (-w_diff) // 2, (-w_diff) // 2 + (-w_diff) % 2
            mask = mask[:,cropping[0]:(m_w - cropping[1])]

        # Clear margins

        m_h, m_w = mask.shape[:2]
        margins = int(margin * target_height), int(margin * target_width)
        cropped = mask[margins[0]:(m_h - margins[0]), margins[1]:(m_w - margins[1])]

        final = np.pad(cropped, ((margins[0], margins[0]), (margins[1], margins[1])), mode = 'constant')

        filename = metadata[i][0]
        cv2.imwrite(root_folder + 'Masks/' + filename[:-4] + 'D.png', 255 * final)

        kernel = np.ones((3,3),np.uint8)
        eroded = np.uint(cv2.erode(np.float32(final), kernel, iterations = 1))

        cv2.imwrite(root_folder + 'Masks/' + filename[:-4] + 'E.png', 255 * eroded)

        masks.append(eroded)

    return(masks)


def detect_fragments(root_folder, model, save_pics, export_json):

    # Runs the whole fragment detection process on parallel threads on all CPU cores.

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    image_folder = root_folder + 'Originals/'
    detection_graph, category_index = load_graph(model)

    filelist = []
    for path in glob.glob(image_folder + "*.jpg") + glob.glob(image_folder + "*.JPG"):
        filelist.append(os.path.basename(path))

    for i in range(len(filelist)):

        if i%size!=rank: continue
        image_np = prepare_image(filelist[i], image_folder)
        out = run_inference_for_single_image(image_np, detection_graph)
        save_annotations(filelist[i], root_folder, out, category_index, insert = True , export = export_json, save_pictures = save_pics)