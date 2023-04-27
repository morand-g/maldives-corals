from functions import *
import os
import requests
import json

def define_paths():
    """define_paths
    This function defines the paths for the Frame_segmentation folder, the folder containing the images, 
    the folder where the masks will be saved and the JSON file containing the annotations.

    Returns:
        str: returns the paths of the folders
    """
    # Chemin du dossier Frame_segmentation
    data_dir = os.getcwd() + "/data/Frame_segmentation"
    
    # Chemin du dossier avec les photos
    pictures_path = os.path.join(data_dir, "unet_p6")
    
    # Chemin où seront enregistrés les masques
    masks_path = os.path.join(data_dir, "annotations", "test_mask")
    
    # Chemin où se trouve le JSON
    json_path = os.path.join(data_dir, "export-v6.json")
    
    print("Path for masks :", masks_path, "\n", "Path json file:", json_path, "\n", "Path folder pictures:", pictures_path, "\n", "Main path :", data_dir)
    
    return data_dir, pictures_path, masks_path, json_path

def load_json(json_path):
    """load_json
    
    Loads data from a JSON file.
    
    Args:
        json_path (str): the path to the JSON file

    Returns:
        array :  a list of items contained in the JSON file
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    items = []
    for raw in data:
        objects = raw
        items.append(raw)

    return items
#appeler cette fonction avec le chemin vers votre fichier JSON
#items = load_json(json_path)


def create_folder(path):
    """create folder
    
    This function checks if a folder already exists at a given path,
    and if it doesn't, it creates it.

    Args:
        path (str): path to the folder where the images are saved
    """
    # Vérifier si le dossier existe déjà sinon le créer
    if not os.path.exists(path):
        os.makedirs(path)
        print("Le dossier a été créé avec succès.")
    else:
        print("Le dossier existe déjà.")

def rename_files(masks_path):
    """rename_files
    
    the function renames all image files 
    in a folder to give them a name without extension.
    
    Args:
        masks_path (str): path for the masks
    """
    masks_path = "./structure/masks/"
    print(masks_path)
    # Boucle sur tous les fichiers dans le dossier
    for file in os.listdir(masks_path):
        # Vérifie que le fichier est un fichier et non un dossier
        if os.path.isfile(os.path.join(masks_path, file)):
            # Vérifie que le fichier a l'extension ".jpg"
            if ".jpg" in file:
                # Construit le nouveau nom de fichier sans l'extension ".jpg"
                new_name = file.replace(".jpg", "")
                # Renomme le fichier en utilisant le nouveau nom
                os.rename(os.path.join(masks_path, file), os.path.join(masks_path, new_name))
