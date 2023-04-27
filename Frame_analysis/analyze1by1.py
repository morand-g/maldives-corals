import sys
from main import *

#########################################################################################
#                                                                                       #
#         This file is a utility that needs to be called from the command line.         #
#           It is designed to run the whole analysis process (or parts of it)           #
#                         on a large number of frames                                   #
#                                                                                       #
#########################################################################################

# Path of the file containing the list of frames to analyze
taglist_path = '/home/mdc/Desktop/SmallFrames.txt'

# Path of the working folder where temporary files will be written
root_folder = '/home/mdc/Desktop/RunningFiles/'

fgvc_model = 'fgvc101-716521'
unet_model = 'unet-28000'

# Number of frames to be analyzed per batch
chunk_size = 3

tag_list = ['INIT1', 'INIT2']

args = sys.argv

if len(args) < 2:
	usecase = 'all'
else:
	usecase = args[1]


# If run in debug mode it will analyze the tag list defined here:
tag_list = ['LG4050', 'LG3366']#'LG1233', 'LG3188', 'LG2029', 'LG3194']

if usecase == 'debug':

	clear_folder(root_folder)
	get_monitoring_pictures([[t] for t in tag_list], root_folder + 'Originals/')

	if args[2] == 'all':
		analyze_frames(root_folder, tag_list, save_pics = True, export_json = False)

	elif usecase == 'filecheck':
		clear_folder(root_folder)
		get_monitoring_pictures([[t] for t in tag_list], root_folder + 'Originals/')

	elif args[2] == 'coral':
		os.system("mpiexec -n 8 python3 parallel_ops.py coral " + ' '.join([root_folder, fgvc_model, 'True', 'False']))

	elif args[2] == 'frames':
		os.system("python3 parallel_ops.py masks " + ' '.join([root_folder, unet_model]))
		os.system("mpiexec -n 8 python3 parallel_ops.py structure " + ' '.join([root_folder, 'True']))

	elif args[2] == 'obs':
		os.system("mpiexec -n 8 python3 parallel_ops.py observations " + ' '.join([root_folder, 'True']))

	elif args[2] == 'frags':
		os.system("mpiexec -n 8 python3 parallel_ops.py fragments " + ' '.join([root_folder]))

	elif args[2] == 'obs+frags':
		os.system("mpiexec -n 8 python3 parallel_ops.py observations " + ' '.join([root_folder, 'True']))
		os.system("mpiexec -n 8 python3 parallel_ops.py fragments " + ' '.join([root_folder]))	

elif usecase == 'update-all':
	tag_list = need_to_update()
	total = len(tag_list)
	chunks = [tag_list[x:x+chunk_size] for x in range(0, total, chunk_size)]
	i = 0
	for tags in chunks:
		update(root_folder, tags)
		i += 1
		print('{} Updated, {} more to go (out of {})'.format(", ".join(tags), len(tag_list) - chunk_size*i, total))

else:
	with open(taglist_path, 'r') as fin:	
		tag_list = [tag.strip() for tag in fin.readlines()]
		total = len(tag_list)


	while(len(tag_list) > 1):
		with open(taglist_path, 'r') as fin:
			
			tag_list = [tag.strip() for tag in fin.readlines()]

			if len(tag_list) >= chunk_size:
				tags = tag_list[:chunk_size]
			else:
				tags = tag_list

			#### Check if files are here
			if usecase == 'filecheck':
				clear_folder(root_folder)
				get_monitoring_pictures([[t] for t in tags], root_folder + 'Originals/')

			#### Run whole process
			elif usecase == 'all':
				analyze_frames(root_folder, tags, save_pics = False, export_json = False)

			#### Just observations
			elif usecase == 'obs':
				clear_folder(root_folder)
				get_monitoring_pictures([[t] for t in tags], root_folder + 'Originals/')
				os.system("mpiexec -n 8 python3 parallel_ops.py observations " + ' '.join([root_folder, 'False']))

			#### Just fragments
			elif usecase == 'frags':
				clear_folder(root_folder)
				get_monitoring_pictures([[t] for t in tags], root_folder + 'Originals/')
				os.system("mpiexec -n 8 python3 parallel_ops.py fragments " + root_folder)
			
			#### Obs + Frags
			elif usecase == 'obs+frags':
				clear_folder(root_folder)
				get_monitoring_pictures([[t] for t in tags], root_folder + 'Originals/')
				os.system("mpiexec -n 8 python3 parallel_ops.py observations " + ' '.join([root_folder, 'False']))
				os.system("mpiexec -n 8 python3 parallel_ops.py fragments " + root_folder)

			### Update after monitoring
			elif usecase == 'update':
				update(root_folder, tags)


			print('{} Done, {} more to go (out of {})'.format(", ".join(tags), max(0, len(tag_list) - chunk_size), total))


			with open(taglist_path, 'w') as fout:
				if len(tag_list) >= chunk_size:
					fout.writelines([t + '\n' for t in tag_list[chunk_size:]])


	print('All frames complete')