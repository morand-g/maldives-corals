from functions import *
from path_creation import *
import os
import requests
import json

def download_masks(items):
    """download_masks
    
    This function downloads mask images, only masks associated with frame_bar are downloaded, 
    and they are saved in folders with the name of the associated object.

    Args:
        items (array): a list of items contained in the JSON file
    """
    for item in items:  
        image_id = item['External ID']

        # Chemin du dossier à créer
        new_folder_path = os.path.join('structure/pictures/', image_id)

        create_folder(new_folder_path)

        i = 0
        # téléchargement seulement des masques frame_bar 
        for obj in item['Label']['objects']:

            if obj['value'] == 'frame_bar':
                i = i + 1
                image_url = obj['instanceURI']
                response = requests.get(image_url)

                if response.status_code == 200:
                    image_content = response.content
                    with open(os.path.join(new_folder_path, f"frame_{i}.png"), "wb") as f:
                        f.write(image_content)
                    print(f"L'image a été enregistrée avec succès : frame_{i}.png")
                    print(item['External ID'])
                else:
                    print("Impossible de télécharger l'image.")
#pour l'utilisé : download_masks(items)

def superpose_masks(pictures_path, masks_path):
    """superpose_masks
    
    the function combines several mask images into one superimposed image,
    which represents the intersection of the masks.
    This image is then saved in a specified folder.

    Args:
        pictures_path (str): path for the pictures
        masks_path (str): path for the masks
    """
    # Définir les noms des dossiers contenant les images à superposer
    image_folders = os.listdir(pictures_path)
    directory = pictures_path

    for image_folder in image_folders:
        # Définir le chemin du dossier contenant les images à superposer
        images_dir = os.path.join(directory, image_folder)

        # Charger les images
        images_png = []
        for i in range(1, 8):
            nom_de_fichier = os.path.join(images_dir, f"frame_{i}.png")
            if os.path.isfile(nom_de_fichier):
                image = Image.open(nom_de_fichier)
                images_png.append(image)

        # Superposer les images
        if images_png:
            image_superposee = images_png[0]
            for i in range(1, len(images_png)):
                image_superposee.alpha_composite(images_png[i])

            # Enregistrer l'image superposée
            mask_dir = masks_path
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            image_superposee.save(os.path.join(mask_dir, f"{image_folder}_mask.png"))

def delete_unmatched_images():
    """delete_unmatched_images
    
    the function deletes the images that do not have a corresponding mask in a dataset, 
    by comparing the file names between the images and the masks.
    
    """
    # Chemin des dossiers contenant les images et les annotations
    images_dir = os.path.join(os.getcwd(), "CORAL_FRAMES")
    masks_path = os.path.join(os.getcwd(), "structure/masks")

    # Obtenir la liste des noms de fichiers d'annotations sans l'extension ".png"
    annotation_files = [os.path.splitext(f)[0] for f in os.listdir(masks_path) if f.endswith(".png")]

    # Parcourir tous les dossiers "SHxxx" contenant les images
    for subdir in os.listdir(images_dir):
        subdir_path = os.path.join(images_dir, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith("SH"):
            # Parcourir tous les fichiers image dans le dossier
            for image_file in os.listdir(subdir_path):
                # Vérifier si le fichier est une image
                if image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".png"):
                    image_name = os.path.splitext(image_file)[0]
                    # Vérifier si le nom du fichier image correspond à un nom de fichier d'annotation
                    if ' ' + image_name in annotation_files:
                        # conserver l'image
                        print("==========================================")
                        print(os.path.join(subdir_path, image_file))
                    else:
                        # supprimer l'image
                        os.remove(os.path.join(subdir_path, image_file))



