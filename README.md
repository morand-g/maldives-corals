# <span id="anchor"></span>Analyzing coral restoration with machine learning

# Table of Contents
* [<span id="anchor-1"></span>Introduction](#<span-id="anchor-1"></span>Introduction)
* [<span id="anchor-7"></span>Training the fragment detection model](#<span-id="anchor-7"></span>Training-the-fragment-detection-model)
* [<span id="anchor-6"></span>Training the frame structure model](#<span-id="anchor-6"></span>Training-the-frame-structure-model)
* [<span id="anchor-2"></span>Frame analysis](#<span-id="anchor-2"></span>Frame-analysis)
	* [<span id="anchor-3"></span>Data storage](#<span-id="anchor-3"></span>Data-storage)
	* [<span id="anchor-4"></span>Working principle](#<span-id="anchor-4"></span>Working-principle)
	* [<span id="anchor-5"></span>Running the algorithms on new frames](#<span-id="anchor-5"></span>Running-the-algorithms-on-new-frames)
* [<span id="anchor-8"></span>Notes](#<span-id="anchor-8"></span>Notes)
	* [<span id="anchor-9"></span>Image editing](#<span-id="anchor-9"></span>Image-editing)
	* [<span id="anchor-10"></span>Frame encoding](#<span-id="anchor-10"></span>Frame-encoding)
	* [<span id="anchor-11"></span>Python requirements](#<span-id="anchor-11"></span>Python-requirements)
* [<span id="anchor-12"></span>Statistics and Results](#<span-id="anchor-12"></span>Statistics-and-Results)



## <span id="anchor-1"></span>Introduction

With hundreds of thousands of monitoring pictures, we have one of the most extensive database on coral restoration in the world. This project’s goal is to extract the information contained within these pictures. We can then analyze this data to understand coral growth and resilience better.

This repository contains all the code we wrote to analyze our coral frames pictures and produce the results for the paper.

## <span id="anchor-7"></span>Training the fragment detection model

All necessary files are in the *Fragment detection* folder.

1.  Create annotations with the [Sloth tool](https://sloth.readthedocs.io/).

6.  Open the *Annotations formatting* jupyter notebook, check the
    filepaths and run all cells. (Except the last *JSON files handling*
    part).

7.  Copy and paste all TFRecords files from the *TFRecords* folder to your training data folder in *data/frags*.

8.  Run the *Start training* bash file.

9.  When the loss stabilizes, terminate training. Check in the
    *models/fgvc* the number of the last checkpoint. Update the
    *Fragment detection/Export model* file with this number.

10. Run the *Export model* bash file

11. Update the model number in *Frame analysis/main.py*

## <span id="anchor-6"></span>Training the frame structure model

All necessary files are in the *Frame segmentation* folder.

1.  Annotate pictures on labelbox.com, and then export and download the json file containing links to the masks.

2.  Open jupyter notebook “Train Unet” and after creating the necessary folders, run the first cell. It might need to be run several times if HTTP errors arise.

3.  Check that all masks are there and run second cell.

4.  Check that we have all pictures and masks in “unet_padded”

5.  In the “Train” cell, manually update the “next step” (0 or last_step + 1). The last_step should be visible at the end of the output, or can be obtained in tensorboard. Then run the cell.

6.  Follow the training by running tensorboard.

7.  When the loss stabilizes, terminate training and export model by copying the three *model.ckpt.\** files to *models_folder/unet-XXXX*. Update name with last step number.

8.  Update the model number in *Frame analysis/main.py*

## <span id="anchor-2"></span>Frame analysis

### <span id="anchor-3"></span>Data storage

All the data is stored in the database, accessible on SEANOE. Seven
different tables contain respectively:

  - **FrameParams**: Camera parameters which define the position of the
    frame on each monitoring picture.
  - **Annotations**: All the coral fragments detected on every
    monitoring picture. This is specific to a frame, a date, and a view.
  - **Observations**: This is the annotations, consolidated for the
    whole frame. It is specific to a frame and a date but not to the
    view. Every observation is linked to one or two annotations.
  - **Fragments**: This is the observations, consolidated for all
    dates. It is specific to a frame, and that’s it. Every fragment is
    linked to one or more observations.
  - **Status** : This is the status of each fragment at each monitoring
    date (starting from first detection)
  - **FSFrames**: This is the list of coral frames.
  - **FSMonitoring**: This is the list of monitoring sessions, with associated picture filenames.

### <span id="anchor-4"></span>Working principle

- **Fragment detection** : We run the fgvc-101 CNN on all pictures to
    detect coral colonies. They will all be classified as Pocillopora,
    Acropora, or Dead coral. It outputs sets of annotations that are
    inserted into the database.
- **Frame detection** : We run the Unet model on all pictures to
    detect the frames. This will output binary masks, that are
    temporarily stored in *unet_masks.npy.*
- **Frame position** : We try random camera parameters to find the
    best match with the masks. The best fit parameters are inserted into
    the database.
- **Observation creation** : We match the annotations with specific
    bars of the frame. We start with the bars in front of each view:
    **H00**: H00, **H03**: H02 & H04, **H06**: H06, **H09**: H08 & H10.
    We then add the potential missing annotations by analyzing secondary
    viewpoints: **H00**: H10 & H02, **H06**: H04 & H08. The resulting
    observations are inserted into the database.
- **Fragment creation** : We analyze all observations from a frame to
    match them and link successive observations of the same fragment.
    The resulting fragments are inserted into the database.

### <span id="anchor-5"></span>Running the algorithms on new frames

All the necessary files are in the *Frame Analysis* folder.

Python functions are divided into 8 files :

  - **sqlfunctions.py** : handles all operations with the database
  - **inference.py** : handles all operations with the deep learning
    models
  - **detectframe.py** : runs the algorithm to calculate the frame
    position
  - **datanalysis.py** : all functions necessary to analyze the contents
    of the frames
  - **main.py** : contains the function to run everything at once
  - **tools.py** : miscellaneous functions for files handling etc.
  - **parallel_ops.py** : main file that should be called from command
    line to run operations
  - **analyze1by1.py** : file that should be called from command
    line to automatically run the algorithm on a large number of frames.

## <span id="anchor-8"></span>Notes

In order to run everything except the *Results* notebook for statistics, there is significant adaptation work to be done, mostly to match the folder paths to your folder structure.

### <span id="anchor-9"></span>Image editing

In this process, we have to edit the pictures, both before training and
before running detection models. We have to be mindful to edit the each
picture **only once**, otherwise the image quality degrades quickly.

As we cannot know which pictures are already edited in the database, we
assume they are not. So some pictures will have been edited twice in the
end.

To prevent overediting, all the pictures are stored unedited, except in
two places:

  - Final pictures used to visualize results may be edited
  - For fragment detection, the folders used to create the TFRecords
    must contain edited files (handled by the annotation formatting
    functions).

### <span id="anchor-10"></span>Frame encoding

The bars are named with successive characters according to the following
rules:

  - V for vertical or H for horizontal
  - Face from 00 to 11. Face 00 is the one with the tag. The next face
    on the left is 02, and so on. Vertical bars are the odd number
    between the two faces they are adjacent to.
  - Letter: A is the lowest bar, B the next one up, and so on.

Some examples : H02B, V07A, H10C

### <span id="anchor-11"></span>Python requirements

We are using Python 3.6 with packages installed with Pip, with a few
exceptions.

  - Tensorflow 1.15 was compiled from source to have the best possible
    performance on this computer.
  - Line 208 was changed in *object_detection/utils/visualization_util.py* to adapt fontsize to image size.
  - *tf_unet* downloaded from [Github](https://github.com/jakeret/tf_unet)

## <span id="anchor-12"></span>Statistics and Results

All the results mentioned in the paper can be calculated from the database using the *Results* jupyter notebook.
To reproduce our results, we recommend using the database we published on SEANOE rather than generating it again, as it takes several weeks. To do this, you will need to setup a SQL server and import our database, and update the credentials in the *Statistics/sql.py* file.