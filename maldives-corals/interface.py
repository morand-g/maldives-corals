"""
This is the interface for the model used to detect
individual coral fragments' positions on a structure.
It is made up of two components:
- an object detection model thats detects the position of individual
coral fragments on an image
- an image segmentation model that detects the structure on the picture
and assigns each fragment to a position on the bars
- a model that merges all the pictures for each structure 
NOTE: this is an abstract class, see model.py for the
implementation.
"""
from abc import ABC, abstractclassmethod
from typing import Iterable

class Model(ABC):

    @abstractclassmethod
    def fit_corals_detection(img: Iterable, annot: Iterable):
        """
        Fits the fragments detection model.
        Args:
            img: iterable of images as matrices (e.g. images opened with PIL)
            annot: iterable containing iterables of annotations, one for each 
            image. Each annotation contains a fragment's category (acropora, 
            bleached, etc...) as well as its bounding box's coordinates on 
            the image.
        
        Returns None.
        """
        pass

    @abstractclassmethod
    def fit_structure_detection(img: Iterable, fragments_location: Iterable):
        """
        Fits the structure detection model, finding out each fragment's
        position on each bar for all structures.
        Args:
            img: iterable where each element contains the images for one structure
            from different angles
            fragments_location: iterable

        """
        pass

    @abstractclassmethod
    def predict(img: list) -> dict:
        """Makes predictions using both models."""
        pass