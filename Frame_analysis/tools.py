import glob
import os
from ftplib import FTP

from shutil import copyfile, copy

from sqlfunctions import get_monitoring_sets

from PIL import Image


ftp_host = ""
ftp_user = ""
ftp_pass = ""


def clear_folder(root_folder):

    # Clear or create all necessary subfolders

    if not(os.path.exists(root_folder)):
        os.mkdir(root_folder)

    if os.path.exists(root_folder + 'Originals'):
        for f in glob.glob(root_folder + 'Originals/*'):
            os.remove(f)
    else:
        os.mkdir(root_folder + 'Originals')

    if os.path.exists(root_folder + 'Annotations'):
        for f in glob.glob(root_folder + 'Annotations/*'):
            os.remove(f)
    else:
        os.mkdir(root_folder + 'Annotations')

    if os.path.exists(root_folder + 'Masks'):
        for f in glob.glob(root_folder + 'Masks/*'):
            os.remove(f)
    else:
        os.mkdir(root_folder + 'Masks')

    if os.path.exists(root_folder + 'Frames'):
        for f in glob.glob(root_folder + 'Frames/*'):
            os.remove(f)
    else:
        os.mkdir(root_folder + 'Frames')

    if os.path.exists(root_folder + 'Summaries'):
        for f in glob.glob(root_folder + 'Summaries/*'):
            os.remove(f)
    else:
        os.mkdir(root_folder + 'Summaries')

    if os.path.exists(root_folder + 'JSON'):
        for f in glob.glob(root_folder + 'JSON/*'):
            os.remove(f)
    else:
        os.mkdir(root_folder + 'JSON')



def get_monitoring_pictures(tagdate_list, image_folder):

    # Copy all monitoring pictures with a tag in tag_list from the diskstation to the image_folder

    diskstation = ''

    for tagdate in tagdate_list:
        tag = tagdate[0]
        resort = 'LG'
        if tag[:2] == 'KH':
            resort = 'KH'

        if len(tagdate) > 1:
            dates = tagdate[1]
        else:
            mon_sets = get_monitoring_sets(tag)
            dates = [m['Date Code'] for m in mon_sets]

        for date in dates:
            date = str(date).zfill(2)
            for i in range(0, 10, 3):
                filename = tag + 'H0' + str(i) + str(date) + '.jpg'

                if resort == 'LG':
                    imgpath = diskstation + 'Unsized/20' + date[:2] + '/' + date[2:4] + '/' + filename
                    if os.path.exists(imgpath):
                        copy(imgpath, image_folder)
                    elif os.path.exists(imgpath.replace('Unsized', 'Resized')):
                        copy(imgpath.replace('Unsized', 'Resized'), image_folder)
                    else:
                        print('Picture {} missing'.format(filename))
                elif resort == 'KH':
                    imgpath = '/media/mdc/Storage/KH_pics/' + filename
                    if not(os.path.exists(imgpath)):
                        os.system("scp " + filename + ' .')
                    copy(imgpath, image_folder)


def sets_from_path(image_folder):

    # Opens pictures path and return all monitoring pictures sets

    picture_sets = {}

    for path in glob.glob(image_folder + "*.jpg"):
        filename = os.path.basename(path)
        tag, date = filename[:6], filename[9:15]
        if tag + date in picture_sets:
            picture_sets[tag + date].append(filename)
        else:
            picture_sets[tag + date] = [filename]

    for key in picture_sets:

        if len(picture_sets[key]) != 4:
            print("Set {}:{} not complete".format(key[:6], key[6:]))

    return(picture_sets)


def tags_from_path(image_folder):

    # Opens pictures path and return all present tags

    tags = []

    for path in glob.glob(image_folder + "*.jpg"):
        filename = os.path.basename(path)
        tag = filename[:6]
        if not (tag in tags):
            tags.append(tag)

    return(tags)


def obs_to_bar(obss):

    # Takes a list of observations and returns a dictionary of them sorted by bar.

    all_obs = {}

    for obs in obss:
        if obs['Bar'] in all_obs:
            all_obs[obs['Bar']].append(obs)
        else:
            all_obs[obs['Bar']] = [obs]

    return(all_obs)


def format_obs(obss):

    # Return pairs of date and corresponding dictionnaries of observations, grouped by bar.

    all_obs = {}

    for obs in obss:
        if obs['Date'] in all_obs:
            if obs['Bar'] in all_obs[obs['Date']]:
                all_obs[obs['Date']][obs['Bar']].append(obs)
            else:
                all_obs[obs['Date']][obs['Bar']] = [obs]

        else:
            all_obs[obs['Date']] = {obs['Bar'] : [obs]}

    
    all_obs_list = [[date, all_obs[date]] for date in all_obs]
    all_obs_list.sort(key = lambda pair: pair[0])

    return(all_obs_list)

    
def color(ref):

    # In BGR

    colors = {}

    colors['H00A'] = (20, 0, 200)
    colors['H00B'] = (20, 100, 220)
    colors['H00C'] = (20, 150, 255)

    colors['H02A'] = (220, 0, 0)
    colors['H02B'] = (255, 70, 70)
    colors['H02C'] = (255, 130, 130)

    colors['H04A'] = (150, 150, 20)
    colors['H04B'] = (200, 200, 20)
    colors['H04C'] = (255, 255, 20)

    colors['H06A'] = (100, 150, 0)
    colors['H06B'] = (50, 200, 0)
    colors['H06C'] = (0, 255, 0)

    colors['H08A'] = (109, 32, 62)
    colors['H08B'] = (213, 109, 145)
    colors['H08C'] = (246, 156, 216)

    colors['H10A'] = (0, 150, 255)
    colors['H10B'] = (0, 200, 255)
    colors['H10C'] = (0, 255, 255)

    return(colors[ref])


def download_pics(monitoring_id, path = 'diskstation'):

    remote_path = ''

    ftp = FTP(ftp_host)
    ftp.set_pasv(False)
    ftp.login(ftp_user, ftp_pass)
    ftp.cwd(remote_path)

    for view in ['H00', 'H03', 'H06', 'H09']:

        filename = monitoring_id[:6] + view + monitoring_id[6:] + '.jpg'
        if path == 'diskstation':
            
            ftp.retrbinary("RETR " + filename, open('/tmp/' + filename, 'wb').write)

            im = Image.open('/tmp/' + filename)
            width, height = im.size
            im.close()

            year = "20" + filename[9:11]
            month = filename[11:13]

            datapics_dir = ''

            if year.isdigit() and month.isdigit():
                if width > 1440:
                    new_path = datapics_dir + 'Unsized/' + year + '/' + month
                else:
                    new_path = datapics_dir + 'Resized/' + year + '/' + month

                if not os.path.exists(new_path): os.makedirs(new_path)

                copyfile('/tmp/' + filename, new_path + '/' + filename)
                print(filename + ' Done')

        else:
            ftp.retrbinary("RETR " + filename, open(path + filename, 'wb').write)
            print(filename + ' Done')

    ftp.quit()