import os
import glob

from tools import *
from inference import *
from detectframe import *
from sqlfunctions import *
from dataanalysis import *

fgvc_model = 'fgvc101-716521'
unet_model = 'unet-28000'

def analyze_frames(root_folder = '/tmp/', tag_list = [], save_pics = False, export_json = False):

	# Runs the whole analysis process
	# With the default parameters, all previous files in root_folder will be overwritten !!

	image_folder = root_folder + 'Originals/'

	# Prepare folder and copy original images

	clear_folder(root_folder)
	get_monitoring_pictures([[t] for t in tag_list], root_folder + 'Originals/')
	print("\nStarting...\n")

	# Detect fragments using deep learning model

	os.system("mpiexec -n 8 python3 parallel_ops.py coral " + ' '.join([root_folder, fgvc_model, str(save_pics), str(export_json)]))
	print("\nCoral detection: Done\n")

	# Detect frame position

	os.system("python3 parallel_ops.py masks " + ' '.join([root_folder, unet_model]))
	os.system("mpiexec -n 8 python3 parallel_ops.py structure " + ' '.join([root_folder, str(save_pics)]))
	print("\nFrame detection: Done\n")

	# Match detections

	os.system("mpiexec -n 8 python3 parallel_ops.py observations " + ' '.join([root_folder, str(save_pics)]))
	print("\nDetections matching: Done\n")

	# Match observations

	os.system("mpiexec -n 8 python3 parallel_ops.py fragments " + root_folder)
	print("\nObservations matching: Done\n")


def update(root_folder, tags):

	# Update database with monitoring pictures that were added after the frames were analyzed

	clear_folder(root_folder)
	get_monitoring_pictures([[tag, get_unanalyzed_sets(tag)] for tag in tags], root_folder + 'Originals/')

	os.system("mpiexec -n 8 python3 parallel_ops.py coral " + ' '.join([root_folder, fgvc_model, 'False', 'False']))
	print("\nCoral detection: Done\n")

	os.system("python3 parallel_ops.py masks " + ' '.join([root_folder, unet_model]))
	os.system("mpiexec -n 8 python3 parallel_ops.py structure " + ' '.join([root_folder, 'False']))
	print("\nFrame detection: Done\n")

	os.system("mpiexec -n 8 python3 parallel_ops.py observations " + ' '.join([root_folder, 'False']))
	print("\nDetections matching: Done\n")

	get_monitoring_pictures([[tag] for tag in tags], root_folder + 'Originals/')
	os.system("mpiexec -n 8 python3 parallel_ops.py fragments " + root_folder)
	print("\nObservations matching: Done\n")