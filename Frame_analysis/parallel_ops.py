import sys
from detectframe import *
from inference import load_pictures_unet, detect_fragments
from dataanalysis import match_annotations, match_observations
from main import *
from mpi4py import MPI

# File to be used from command line
# Mandatory parameter can be any of the following:
# coral				Runs the fragment detection
# masks 			Runs Unet detection on pictures and saves masks
# structure			Runs the frame structure detection
# detections 		Matches annotations to create observations
# observations 		Matches observations to create fragments
# all				Performs all these steps

_, usecase = sys.argv[:2]

if usecase is None:
	print('Which operation needs to be performed?')

if usecase == 'coral':
	_, _, root_folder, model, save_pics, export_json = sys.argv[:6]
	export_json = (export_json == 'True')
	save_pics = (save_pics == 'True')

	detect_fragments(root_folder, model, save_pics, export_json)

if usecase == 'masks':
	_, _, root_folder, model = sys.argv[:4]

	size = MPI.COMM_WORLD.Get_size()

	if size > 1:
		print("Run again on one core only")
	else:
		image_folder = root_folder + 'Originals/'
		pics, metadata = load_pictures_unet(image_folder, target_width = 200)
		unet_masks = get_unet_masks(model, pics, root_folder, metadata, chunk_size = 10, target_width = 200)
		np.save(root_folder + 'unet_masks.npy', unet_masks)

if usecase == 'structure':
	_, _, root_folder, save_pics = sys.argv[:4]

	save_pics = (save_pics == 'True')
	image_folder = root_folder + 'Originals/'

	_, metadata = load_pictures_unet(image_folder, target_width = 200)

	unet_masks = np.load(root_folder + 'unet_masks.npy')

	compute_camera_params(root_folder, unet_masks, metadata)

if usecase == 'observations':
	_, _, root_folder, save_pics = sys.argv[:4]
	save_pics = (save_pics == 'True')
	match_annotations(root_folder, save_pics)

if usecase == 'fragments':
	_, _, root_folder = sys.argv[:3]
	image_folder = root_folder + 'Originals/'
	match_observations(image_folder)

if usecase == 'all':
	if len(sys.argv) > 2:
		root_folder = sys.argv[2]
	else:
		root_folder = '/tmp/'

	if os.path.exists(root_folder + 'taglist.txt'):
		f = open(root_folder + 'taglist.txt', 'r')
		tag_list = [tag.strip() for tag in f.readlines()]

	else:
		tag_list = tags_from_path(root_folder + 'Originals/')

	analyze_frames(root_folder,  tag_list, save_pics = True, export_json = True)