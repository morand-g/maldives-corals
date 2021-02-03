import os

from math import ceil
from copy import deepcopy

from sqlfunctions import *
from itertools import product

import numpy as np
from datetime import date, datetime
from mpi4py import MPI

import cv2

from tools import *
from detectframe import cloud_transform


#################################################################################################
#                                                                                               #
# This file contains all the needed functions to process the data resulting from image analysis #
#                     A description is included in each function definition                     #
#                                                                                               #
#################################################################################################


# Frames which need to be rotated relative to their initial positions, because of monitoring mistakes.
rotated_frames = {'LG1716': ['200925', 6], 
'LG1347': ['190925', 2], 
'LG3194': ['200609', -2],
'LG3366': ['200924', 2],
'LG3188': ['190219', -4]}


class bar:

    # This class is meant to represent a bar of the frame in a specific picture.
    # Its default attributes are its 3D coordinates
    # The project method creates 2d coordinates with the given parameters
    # The distance method returns the shortest distance between a point and this bar, in the 2D plane

    def __init__(self, xy):

        # Instanciate a new bar with 3d coordinates
        self.x1_3d, self.y1_3d, self.z1_3d, self.x2_3d, self.y2_3d, self.z2_3d = xy

    def __repr__(self):

        # Returns 2d coordinates if the bar has already been projected
        if(hasattr(self,'x1')):
            return('<projected bar: x1 = {}, y1 = {} | x2 = {}, y2 = {}>'.format(self.x1, self.y1, self.x2, self.y2))
        else:
            return('<unprojected bar>')

    def project(self, params, imsize, v):

        # Projects 3d bar on picture plane.

        pts = np.array([[self.x1_3d, self.y1_3d, self.z1_3d],[self.x2_3d, self.y2_3d, self.z2_3d]])
        M = cloud_transform(params[0], [params[1], params[2], params[3]], view = v, points = pts)

        h, w, _ = imsize

        Timage = [w  * params[5], h * params[6]]
        scaling = w * params[4]

        self.x1, self.y1 = -scaling * M[0][0] + Timage[0], scaling * M[0][2] + Timage[1]
        self.x2, self.y2 = -scaling * M[1][0] + Timage[0], scaling * M[1][2] + Timage[1]

        self.x1, self.y1 = self.x1 / w, self.y1 / h
        self.x2, self.y2 = self.x2 / w, self.y2 / h

        self.a = (self.y2 - self.y1) / (self.x2 - self.x1)
        self.b = self.y1 - self.a * self.x1
        if(self.x2 < self.x1):
            x, y = self.x1, self.y1
            self.x1, self.y1 = self.x2, self.y2
            self.x2, self.y2 = x, y


    def pointdistance(self, xi, yi):

        # Returns distance between specified point and bar

        bi = yi - self.a * xi
        projxi = xi + (bi - self.b) * self.a / (1 + self.a**2)
        projyi = projxi * self.a + self.b

        if(projxi >= min(self.x1, self.x2) and projxi <= max(self.x1, self.x2)):
            d = np.sqrt((projyi - yi)**2 + (projxi - xi)**2)
            return(d)
        else:
            d1 = np.sqrt((self.y1 - yi)**2 + (self.x1 - xi)**2)
            d2 = np.sqrt((self.y2 - yi)**2 + (self.x2 - xi)**2)
            if(d1 < d2):
                return(3 * d1)
            else:
                return(3 * d2)


    def distance(self, TLx, TLy, width, height):

        # If bar and fragment don't overlap, returns shortest distance
        # If they do, returns the approximate negative area of overlap (so the minimum is the best fit)
        # We have to separate the cases depending on the orientation of the bar
        # Also returns best position of fragment on bar
        
        Lx, Mx, Rx = TLx, TLx + width / 2, TLx + width
        Ty, My, By = TLy, TLy + height / 2, TLy + height

        bmax = max(self.pointdistance(Lx, By), self.pointdistance(Rx, By))
        tmax = max(self.pointdistance(Lx, Ty), self.pointdistance(Rx, Ty))

        if self.a < -0.01:

            #if (By >= self.a * Mx + self.b + 0.2 * height and Ty <= self.a * Mx + self.b - 0.2 * height) or (height < 0.1 and (bmax < 0.03 or tmax < 0.015 )) :
            if (By >= self.a * Rx + self.b + 0.1 * height and Ty <= self.a * Lx + self.b - 0.3 * height) or (height < 0.1 and (bmax < 0.025 or tmax < 0.015 )) :
                
                # Orthogonal Overlap
                
                bTR = Ty - self.a * Rx
                projTRx = Rx + (bTR - self.b) * self.a / (1 + self.a**2)
                relativeProjTR = (projTRx - self.x1) / (self.x2 - self.x1)

                bBL = By - self.a * Lx
                projBLx = Lx + (bBL - self.b) * self.a / (1 + self.a**2)
                relativeProjBL = (projBLx - self.x1) / (self.x2 - self.x1)

                if relativeProjBL * (relativeProjTR - 1) < 0:
                    d = -1
                elif relativeProjBL < 0 and relativeProjTR > 0:
                    d = - relativeProjTR / (relativeProjTR - relativeProjBL)
                    if d > -0.6:
                        d = -d
                elif relativeProjBL < 1 and relativeProjTR > 1:
                    d = - (1 - relativeProjBL) / (relativeProjTR - relativeProjBL)
                    if d > -0.6:
                        d = -d
                else:
                    d = min(abs(relativeProjBL - 1), abs(relativeProjTR))

                return(d, np.clip(0.5 * (relativeProjBL + relativeProjTR), 0, 1))

            else:

                # No overlap
                return(1, 0)


        elif self.a > 0.01:

            #if (By >= self.a * Mx + self.b + 0.2 * height and Ty <= self.a * Mx + self.b - 0.2 * height) or (height < 0.1 and (bmax < 0.03 or tmax < 0.015 )) :
            if (By >= self.a * Lx + self.b + 0.1 * height and Ty <= self.a * Rx + self.b - 0.3 * height) or (height < 0.1 and (bmax < 0.025 or tmax < 0.015 )) :

                # Orthogonal Overlap

                bTL = Ty - self.a * Lx
                projTLx = Lx + (bTL - self.b) * self.a / (1 + self.a**2)
                relativeProjTL = (projTLx - self.x1) / (self.x2 - self.x1)

                bBR = By - self.a * Rx
                projBRx = Rx + (bBR - self.b) * self.a / (1 + self.a**2)
                relativeProjBR = (projBRx - self.x1) / (self.x2 - self.x1)

                if relativeProjTL * (relativeProjBR - 1) < 0:
                    d = -1
                elif relativeProjTL < 0 and relativeProjBR > 0:
                    d = - relativeProjBR / (relativeProjBR - relativeProjTL)
                    if d > -0.6:
                        d = -d
                elif relativeProjTL < 1 and relativeProjBR > 1:
                    d = - (1 - relativeProjTL) / (relativeProjBR - relativeProjTL)
                    if d > -0.6:
                        d = -d
                else:
                    d = min(abs(relativeProjTL), abs(relativeProjBR - 1))

                return(d, np.clip(0.5 * (relativeProjTL + relativeProjBR), 0, 1))

            else:

                # No overlap
                return(1, 0)

        else:

            if (By >= self.y1 + 0.1 * height and Ty <= self.y1 - 0.3 * height) or (height < 0.1 and (bmax < 0.025 or tmax < 0.015 )) :

                # Orthogonal Overlap

                relativeProjTL = (Lx - self.x1) / (self.x2 - self.x1)
                relativeProjBR = (Rx - self.x1) / (self.x2 - self.x1)

                if relativeProjTL * (relativeProjBR - 1) < 0:
                    d = -1
                elif relativeProjTL < 0 and relativeProjBR > 0:
                    d = - relativeProjBR / (relativeProjBR - relativeProjTL)
                    if d > -0.6:
                        d = -d
                elif relativeProjTL < 1 and relativeProjBR > 1:
                    d = - (1 - relativeProjTL) / (relativeProjBR - relativeProjTL)
                    if d > -0.6:
                        d = -d
                else:
                    d = min(abs(relativeProjTL - 1), abs(relativeProjBR))

                return(d, np.clip(0.5 * (relativeProjTL + relativeProjBR), 0, 1))

            else:

                # No overlap
                return(1, 0)     


def load_bars(view):

    # Returns a default dictionary of the bars we expect from this view.
    # The frame is turned so the bars' references for each view represent the absolute references of the 3D frame bars.

    bars = np.loadtxt('/home/mdc/Desktop/Frame analysis/SmallFrameLines.csv', skiprows = 1, delimiter = ',', usecols = [0, 1, 2, 3, 4, 5])


    if view == "H00":
        hbars = {'H02A': bar(bars[7]), 'H02B': bar(bars[8]), 'H02C': bar(bars[9]),\
            'H00A': bar(bars[11]), 'H00B': bar(bars[12]), 'H00C': bar(bars[13]),\
            'H10A': bar(bars[15]), 'H10B': bar(bars[16]), 'H10C': bar(bars[17])}
        vbars = {'V09A': bar(bars[18]), 'V09B': bar(bars[4]), 'V11A': bar(bars[14]),\
            'V11B': bar(bars[5]), 'V01A': bar(bars[10]), 'V01B': bar(bars[0]),\
            'V03A': bar(bars[6]), 'V03B': bar(bars[1])}
    elif view == "H03":
        hbars = {'H04A': bar(bars[11]), 'H04B': bar(bars[12]), 'H04C': bar(bars[13]),\
            'H02A': bar(bars[15]), 'H02B': bar(bars[16]), 'H02C': bar(bars[17])}
        vbars = {'V05A': bar(bars[10]), 'V05B': bar(bars[0]), 'V01A': bar(bars[18]),\
            'V01B': bar(bars[4]), 'V03A': bar(bars[14]), 'V03B': bar(bars[5])}
    elif view == "H06":
        hbars = {'H08A': bar(bars[7]), 'H08B': bar(bars[8]), 'H08C': bar(bars[9]),\
            'H06A': bar(bars[11]), 'H06B': bar(bars[12]), 'H06C': bar(bars[13]),\
            'H04A': bar(bars[15]), 'H04B': bar(bars[16]), 'H04C': bar(bars[17])}
        vbars = {'V09A': bar(bars[6]), 'V09B': bar(bars[1]), 'V07A': bar(bars[10]),\
            'V07B': bar(bars[0]), 'V05A': bar(bars[14]), 'V05B': bar(bars[5]),\
            'V03A': bar(bars[18]), 'V03B': bar(bars[4])}
    elif view == "H09":
        hbars = {'H10A': bar(bars[11]), 'H10B': bar(bars[12]), 'H10C': bar(bars[13]),\
            'H08A': bar(bars[15]), 'H08B': bar(bars[16]), 'H08C': bar(bars[17])}
        vbars = {'V07A': bar(bars[18]), 'V07B': bar(bars[4]), 'V09A': bar(bars[14]),\
            'V09B': bar(bars[5]), 'V11A': bar(bars[10]), 'V11B': bar(bars[0])}

    return(hbars, vbars)


def superposition(annot0, annot1):

    # Returns the proportion of overlap between two annotations.

    annot0np = np.zeros((1000, 1000), np.int8)
    annot1np = np.zeros((1000, 1000), np.int8)

    annot0x1 = (int(1000*annot0['TopLeftX']), int(1000*annot0['TopLeftY']))
    annot0x2 = (int(1000*(annot0['TopLeftX'] + annot0['Width'])), int(1000*(annot0['TopLeftY'] + annot0['Height'])))

    annot1x1 = (int(1000*annot1['TopLeftX']), int(1000*annot1['TopLeftY']))
    annot1x2 = (int(1000*(annot1['TopLeftX'] + annot1['Width'])), int(1000*(annot1['TopLeftY'] + annot1['Height'])))

    cv2.rectangle(annot0np, annot0x1, annot0x2, 1, -1)
    cv2.rectangle(annot1np, annot1x1, annot1x2, 1, -1)

    return(np.sum(annot0np * annot1np) / np.sum(np.maximum(annot0np, annot1np)))


def extrapolate_coords(obs, view, ref, bar, scale_factor):

    # Calculates approximate coordinates to display fragments on other pictures than the one it was detected on.

    spot = obs[0]
    annot = obs[-1][0]

    offsetw = 0
    offseth = 0

    if view in ('H00', 'H06'):

        # Distort perspective to display the fragments where they would appear on the picture

        if ref[:3] in ('H02', 'H08'):
            spot = spot**2
            offsetw, offseth = 0.04, -0.03
        else:
            spot = 1-(1-spot)**2
            offsetw, offseth = -0.04, -0.03

    else:
        if ref[:3] in ('H04', 'H10'):
            offsetw, offseth = 0.03, -0.03
        else:
            offsetw, offseth = -0.03, -0.03

    if type(obs) is dict:
        fh, fw = scale_factor * obs['AdjHeight'], scale_factor * obs['AdjWidth']
    else:
        fh, fw = scale_factor * obs[4], scale_factor * obs[5]

    root = bar.x1 + spot * (bar.x2 - bar.x1), bar.y1 + spot * (bar.y2 - bar.y1)
    TLx, TLy = root[0] - (0.5 - offsetw) * fw, root[1] - (0.5 - offseth) * fh
    BRx, BRy = root[0] + (0.5 + offsetw) * fw, root[1] + (0.5 + offseth) * fh

    return(TLx, TLy, BRx, BRy)


def is_foreground(root_folder, image_name, bar, params, annot):

    # Checks if the detected fragment is in the foreground, by verifying whether its assigned bar is visible.
    # Unused as of now because of yielding too many fake negatives.

    if os.path.exists(root_folder + 'Masks/' + image_name[:-4] + 'F.png'):
        mask_path = root_folder + 'Masks/' + image_name[:-4] + 'F.png'
    else:
        mask_path = root_folder + 'Masks/' + image_name[:-4] + 'E.png'

    mask = np.floor(cv2.imread(mask_path, 0) / 255)
    h, w = mask.shape[:2]

    xs = int(w * annot['TopLeftX']), int(w * (annot['TopLeftX'] + annot['Width']))
    ys = int(h * annot['TopLeftY']), int(h * (annot['TopLeftY'] + annot['Height']))
    
    radius = -0.02 * params[4] * w * params[0]

    barnp = np.zeros((h, w))
    cv2.line(barnp, (int(w * bar.x1), int(h * bar.y1)), (int(w * bar.x2), int(h * bar.y2)), 1, np.int(radius))

    annotnp = np.zeros((h, w))
    cv2.rectangle(annotnp, (xs[0], ys[0]), (xs[1], ys[1]), 1, -1)

    result = np.sum(mask * barnp * annotnp) / (np.sum(barnp * annotnp) + 1)

    ifsmall = - (annot['Width'] + annot['Height']) / (0.01 * params[4] * params[0])
    
    return(result < 0.5 or ifsmall < 30)


def match_spots(spotlist1, spotlist2):

    # Takes two lists of spots on a bar (between 0 and 1) and matches them the best possible way

    elementwise_possibilities = [[[] for x in spotlist2] for x in spotlist1]

    for i1 in range(len(spotlist1)):
        for i2 in range(len(spotlist2)):
            if abs(spotlist1[i1] - spotlist2[i2]) < 0.3:
                elementwise_possibilities[i1].append([i1, i2])

    all_possibilities = product(*elementwise_possibilities)
    
    min_score = 10
    min_p = None

    for p in all_possibilities:
        used2 = []
        score = 0
        cancel = False
        for c in p:
            if not(cancel):
                if(len(c)):
                    if c[1] in used2:
                        # spot2 attributed twice
                        cancel = True
                    else:
                        score += abs(spotlist2[c[1]] - spotlist1[c[0]])
                        used2.append(c[1])
                else:
                    score += 1

        if not(cancel) and score < min_score:
            min_score = score
            min_p = p

    return(min_p)


def is_matching(frags1, frags2):

    ############ Used for frame matching, not for frame analysis ############
    # Checks if bar with frags1 can be an older version of bar with frags2
    # Higher score means better fit
    # Unused as of now


    if len(frags1) and len(frags2):
        spots1 = list(map(lambda x: x['Position'], frags1))
        spots2 = list(map(lambda x: x['Position'], frags2))
        correspondances = match_spots(spots1, spots2)
        matches_count = sum([len(c) / 2 for c in correspondances])
        new_frags = len(spots2) - matches_count
        missing_frags = len(spots1) - matches_count
        score = 0
        for c in correspondances:
            if len(c):
                obs1, obs2 = frags1[c[0]], frags2[c[1]]
                date2 = datetime.strptime('20' + str(obs1['Date']).zfill(6), '%Y%m%d').date()
                date1 = datetime.strptime('20' + str(obs2['Date']).zfill(6), '%Y%m%d').date()
                exp_gr = 1.5**((date2 - date1).days / 365)
                if 0.8 * exp_gr * obs1['AdjWidth'] <= obs2['AdjWidth'] <= 1.2 * exp_gr * obs1['AdjWidth'] and 0.8 * exp_gr * obs1['AdjHeight'] <= obs2['AdjHeight'] <= 1.2 * exp_gr * obs1['AdjHeight']:
                    if obs1['Type'] == obs2['Type']:
                        score += 1
                    elif obs2['Type'] == 'Dead':
                        score += 0.7
                    else:
                        score += 0.2
                else:
                    score -= 0.2
    elif len(frags1):
        new_frags = 0
        missing_frags = len(frags1)
    elif len(frags2):
        new_frags = len(frags2)
        missing_frags = 0
    else:
        score = 3

    score -= 0.9 * new_frags
    score -= 0.6 * missing_frags

    return(score)


def asym_simi(pop_bars1, pop_bars2):

    # Similarity score between two successive observations sets.
    # Unused as of now
    
    bar_refs = [a + b for a in ['H00', 'H02', 'H04', 'H06', 'H08', 'H10'] for b in ['A', 'B', 'C']]
    
    matching_bars = 0
    for ref in bar_refs:
        if ref in pop_bars2:
            if ref in pop_bars1:
                matching_bars += is_matching(pop_bars1[ref], pop_bars2[ref])
            else:
                matching_bars -= 0.5 * len(pop_bars2[ref])
        else:
            if ref in pop_bars1:
                matching_bars -= 0.2 * len(pop_bars1[ref])
            else:
                matching_bars += 3
    
    return matching_bars


def match_annotations(root_folder, save_pics = True):

    # Matches annotations to create observations

    image_folder = root_folder + 'Originals/'
    picture_sets = sets_from_path(image_folder)

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    keys = list(picture_sets.keys())
    for i in range(len(keys)):     
        if i%size!=rank: continue

        key = keys[i]
        tag, date = key[:6], key[6:]
        clear_observations(tag, date)
        annots, params = {}, {}
        if len(picture_sets[key]) == 4:
            for filename in picture_sets[key]:    
                view = filename[6:9]
                annots[view] = get_annotations(filename)
                params[view] = get_params(filename)

            obs = compute_observations(tag, date, params, annots, root_folder)
            if tag in rotated_frames and date >= rotated_frames[tag][0]:
                obs = rotate_obs(obs, rotated_frames[tag][1])
                print('Rotating observations by {}hrs for set {}:{}'.format(-rotated_frames[tag][1], tag, date))

            save_observations(tag, date, obs)
            if save_pics:
                save_summaries(tag, date, obs, params, root_folder)


def rotate_obs(obs, offset):

    # Rotates the frame so all observations stay on a consistent bar. To account for monitoring mistakes

    new_obs = {}
    for ref in obs:
        new_obs['H' + str((int(ref[1:3]) - offset) % 12).zfill(2) + ref[3]] = obs[ref]

    return(new_obs)


def compute_observations(tag, date, params, annots, root_folder):

	# Takes all annotations of a specific tag and a specific date and returns the corresponding observations

    pop_bars = {}
    scale_factor = {}
    prime_angles = {'H00' : ['H00'], 'H03' : ['H02', 'H04'], 'H06' : ['H06'], 'H09' : ['H08', 'H10']}
    secondary = []

    for view in ['H00', 'H03', 'H06', 'H09'] :

        image_path = root_folder + 'Originals/' + tag + view + date + '.jpg'
        img = cv2.imread(image_path)
        imsize = img.shape
        scale_factor[view] = -0.01 * params[view][0] * params[view][4]

        hbars, vbars = load_bars(view)
        for ref in hbars:
            if ref not in pop_bars:
                pop_bars.update({ref: []})
            hbars[ref].project(params[view], imsize, view)
        
        for annot in annots[view]:
            d = ['', 0]
            for ref in hbars.keys():
                dist, spot = hbars[ref].distance(annot['TopLeftX'], annot['TopLeftY'], annot['Width'], annot['Height'])
                if(dist < d[1]):
                    d = [ref, dist, spot, hbars[ref]]

            if len(d) > 2: 
                if d[0][:3] in prime_angles[view]:

                    pop_bars[d[0]].append([d[2], tag, date, annot['Type'], annot['Height'] / scale_factor[view], annot['Width'] / scale_factor[view], [annot]])
                else:
                    if d[0][:3] in ['H02', 'H08']:
                        d[2] = np.sqrt(d[2])
                    elif d[0][:3] in ['H04', 'H10']:
                        d[2] = 1 - np.sqrt(1-d[2])
                    #if is_foreground(root_folder, tag + view + date + '.jpg', hbars[d[0]], params[view], annot):
                    secondary.append([tag, date, '', d[0], d[2], annot, d[3]])

    for sec in secondary:
        merge_into_observations(sec[0], sec[1], pop_bars, sec[3], sec[4], sec[5], sec[6], scale_factor)

    return(pop_bars)


def merge_into_observations(tag, date, frame_dic, ref, spot, annot, bar, scale_factor):

    # Inserts the given annotation into the referenced observations dictionary
    # This function must only be used on secondary observations.

    angle, bar_width = np.arctan(bar.a), bar.x2 - bar.x1

    if ref[:2] in ('H10', 'H02'):
        view = 'H00'
    else:
        view = 'H06'

    if len(frame_dic[ref]) == 0:
        frame_dic[ref].append([spot, tag, date, annot['Type'], annot['Height'] / scale_factor[view], annot['Width'] / scale_factor[view], [annot]])
    else:
        threshold = {'A' : 0.2, 'B' : 0.3, 'C' : 1}
        max_area = 0.2
        max_index = 0
        min_dist = threshold[ref[3]]
        min_index = 0 

        for i in range(len(frame_dic[ref])):
            obs = frame_dic[ref][i]

            TLx, TLy, BRx, BRy = extrapolate_coords(obs, view, ref, bar, scale_factor[view])
            transposed_annot = {'TopLeftX' : TLx, 'TopLeftY' : TLy, 'Width' : BRx - TLx, 'Height' : BRy - TLy}
 
            if abs(obs[0] - spot) < min_dist:
                min_dist = abs(obs[0] - spot)
                min_index = i
            d = superposition(annot, transposed_annot)
            if d > max_area:
                max_area = d
                max_index = i
        
        if max_area > 0.2:
            frame_dic[ref][max_index][-1].append(annot)      
        elif min_dist < threshold[ref[3]]:
            frame_dic[ref][min_index][-1].append(annot)
        else:
            frame_dic[ref].append([spot, tag, date, annot['Type'], annot['Height'] / scale_factor[view], annot['Width'] / scale_factor[view], [annot]])


def save_summaries(tag, date, obs, params, root_folder):

    # Takes observations and frame parameters and produces a picture with frame bars and associated observations.

    for view in ['H00', 'H03', 'H06', 'H09'] :

        image_path = root_folder + 'Originals/' + tag + view + date + '.jpg'
        img = cv2.imread(image_path)
        imsize = img.shape
        h, w, _ = img.shape

        pop_bars = {}
        hbars, vbars = load_bars(view)
        radius = -0.005 * params[view][0] * params[view][4] * w

        for ref in hbars:
            pop_bars.update({ref: []})
            hbars[ref].project(params[view], imsize, view)

        for ref in vbars:
            vbars[ref].project(params[view], imsize, view)
            cv2.line(img, (int(w*vbars[ref].x1), int(h*vbars[ref].y1)), (int(w*vbars[ref].x2), int(h*vbars[ref].y2)), (255, 255, 255), ceil(radius))

        for ref in pop_bars:
            col = color(ref)
            
            x2, y2 = hbars[ref].x1, hbars[ref].y1

            if ref in obs:

                obs[ref].sort(key=lambda x: x[0])

                for o in obs[ref]:

                    # Is the observation actually from this picture?

                    annot = o[-1][0]

                    if o[-1][0]['MonitoringPicture'][6:9] == view:
                        annot = o[-1][0]
                        picture_shift = False
                    elif len(o[-1]) > 1 and o[-1][1]['MonitoringPicture'][6:9] == view:
                        annot = o[-1][1]
                        picture_shift = False
                    else:
                        picture_shift = True

                    # If not, we transpose approximate coordinates from the other picture

                    if picture_shift:

                        scale_factor = -0.01 * params[view][0] * params[view][4]
                        TLx, TLy, BRx, BRy = extrapolate_coords(o, view, ref, hbars[ref], scale_factor)

                        x1 = TLx

                    else:
                        x1 = annot['TopLeftX']
                        TLx, TLy = x1, annot['TopLeftY']
                        BRx, BRy = x1 + annot['Width'], TLy + annot['Height']

                    y1 = hbars[ref].a * x1 + hbars[ref].b

                    # x1, y1 are the coordinates where the frame bar will stop, on the left of this observation
                    # x2, y2 are the coordinates where the frame bar will start, on the right of the last observation
                    
                    if hbars[ref].a == 0:
                        hbars[ref].a = 0.01

                    # Do we need to connect to the top / bottom of the fragment bounding box instead of the left?

                    if y1 < TLy:
                        y1 = TLy
                        x1 = (y1 - hbars[ref].b) / hbars[ref].a

                    elif y1 > BRy:
                        y1 = BRy
                        x1 = (y1 - hbars[ref].b) / hbars[ref].a
                        if x1 > hbars[ref].x2:
                            x1, y1 = hbars[ref].x2, hbars[ref].y2
                    
                    # Check that the two points are inside the frame segment (errors can arise when a is very small)

                    x1 = np.clip(x1, hbars[ref].x1, hbars[ref].x2)
                    x2 = np.clip(x2, hbars[ref].x1, hbars[ref].x2)
                    y1, y2 = hbars[ref].a * x1 + hbars[ref].b, hbars[ref].a * x2 + hbars[ref].b

                    if(x1 > x2):
                        cv2.line(img, (int(w*x1), int(h*y1)), (int(w*x2), int(h*y2)), col, ceil(radius))
                    
                    # Draw bounding box

                    pad = int(0.01 * h)
                    cv2.line(img, (int(w*TLx), int(h*TLy)), (int(w*BRx), int(h*TLy)), col, pad)
                    cv2.line(img, (int(w*BRx), int(h*TLy)), (int(w*BRx), int(h*BRy)), col, pad)
                    cv2.line(img, (int(w*BRx), int(h*BRy)), (int(w*TLx), int(h*BRy)), col, pad)
                    cv2.line(img, (int(w*TLx), int(h*BRy)), (int(w*TLx), int(h*TLy)), col, pad)


                    cv2.putText(img, annot['Type'], (int(w*TLx), int(h*BRy) + 4*pad), cv2.FONT_HERSHEY_DUPLEX, h/1000, col, 1)
                    cv2.putText(img, str(annot['AnnotationId']), (int(w*TLx), int(h*BRy) + 8*pad), cv2.FONT_HERSHEY_DUPLEX, h/1000, col, 1)
                    
                    x2 = BRx
                    y2 = hbars[ref].a * x2 + hbars[ref].b

                    # Do we need to connect the next frame bar to the top / bottom of the fragment bounding box instead of the right?

                    if y2 < TLy:
                        y2 = TLy
                        x2 = (y2 - hbars[ref].b) / hbars[ref].a

                    elif y2 > BRy:
                        y2 = BRy
                        x2 = (y2 - hbars[ref].b) / hbars[ref].a
                        if x2 < hbars[ref].x1:
                            x2, y2 = hbars[ref].x1, hbars[ref].y1

            # Draw last bar?

            x1, y1 = hbars[ref].x2, hbars[ref].y2

            if(x1 > x2):

                cv2.line(img, (int(w*x2), int(h*y2)), (int(w*x1), int(h*y1)), col, ceil(radius))

                #cv2.putText(img, ref, (int(w*x2),int(h*y2)-50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

        cv2.imwrite(root_folder + 'Summaries/' + tag + view + date + '.jpg', img)


def match_observations(image_folder):

    # Matches observations to create fragments

    tags = tags_from_path(image_folder)

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    for i in range(len(tags)):
        if i%size!=rank: continue

        tag = tags[i]
        clear_fragments(tag)
        
        dates = get_monitoring_dates(tag)
        datesequence = {}
        for i in range(len(dates)):
            datesequence[dates[i]] = i

        observations = format_obs(get_observations(tag))

        frags = compute_fragments(tag, datesequence,  observations)
        fix_frags_type(frags)
        frag_ids = save_fragments(tag, [frag for bar in frags for frag in frags[bar]])
        frag_dates = make_statuses(tag, frag_ids, [frag for bar in frags for frag in frags[bar]])
        update_frags(frag_ids, frag_dates)


def compute_fragments(tag, dates, observations):

    # Takes all observations of a specific tag and returns the corresponding fragments
    # Working recursively by adding observations to fragments chronologically

    if len(observations) == 0:
        return([])
    elif len(observations) == 1:
        date = observations[0][0]
        frags = {}
        for ref in observations[0][1]:
            for obs in observations[0][1][ref]:
                if obs['Bar'] in frags:
                    frags[obs['Bar']].append([obs['Tag'], obs['Bar'], obs['Position'], obs['Type'], [obs]])  
                else:
                    frags[obs['Bar']] = [[obs['Tag'], obs['Bar'], obs['Position'], obs['Type'], [obs]]]
        return(frags)
    else:
        date = observations[-1][0]
        obs_to_insert = observations[-1][1]
        observations = observations[:-1]
        return(merge_into_fragments(dates, obs_to_insert, compute_fragments(tag, dates, observations)))


def merge_into_fragments(dates, obss, frag_dict):

    # Takes all observations from a specific date and adds them to the fragments

    bar_refs = [a + b for a in ['H00', 'H02', 'H04', 'H06', 'H08', 'H10'] for b in ['A', 'B', 'C']]

    for ref in bar_refs:
        if ref in obss:
            if ref in frag_dict:
                spotsf = list(map(lambda x: x[2], frag_dict[ref]))
                spotso = list(map(lambda x: x['Position'], obss[ref]))
                correspondances = match_spots(spotsf, spotso)
                inserted = []
                
                for c in correspondances:
                    if len(c):
                        #Check for size problem
                        size_ratio = obss[ref][c[1]]['AdjHeight'] * obss[ref][c[1]]['AdjWidth'] / (frag_dict[ref][c[0]][-1][-1]['AdjHeight'] * frag_dict[ref][c[0]][-1][-1]['AdjWidth'])
                        if size_ratio > 0.8 :
                            date2 = obss[ref][c[1]]['Date']
                            date1 = frag_dict[ref][c[0]][-1][-1]['Date']
                            if dates[date2] - dates[date1] <= 2:
                                frag_dict[ref][c[0]][-1].append(obss[ref][c[1]])
                                inserted.append(c[1])
                for i in range(len(obss[ref])):
                    if i not in inserted:
                        frag_dict[ref].append([obss[ref][i]['Tag'], obss[ref][i]['Bar'], obss[ref][i]['Position'], obss[ref][i]['Type'], [obss[ref][i]]])

            else:
                frag_dict[ref] = []
                for obs in obss[ref]:
                    frag_dict[ref].append([obs['Tag'], obs['Bar'], obs['Position'], obs['Type'], [obs]])

    return(frag_dict)


def fix_frags_type(frags):

    # Finds the actual fragment genus (most frequent from linked observations)

    for bar in frags:
        for frag in frags[bar]:
            type_dict = {'Pocillopora' : 0, 'Acropora' : 0, 'Frame Tag' : 0}
            for obs in frag[-1]:
                if obs['Type'] in type_dict:
                    type_dict[obs['Type']] += 1

            ordered_categories = [it[0] for it in sorted(type_dict.items(), key=lambda x: x[1])]
            right = ordered_categories[-1]
            if type_dict[right]:
                frag[3] = right
            else:
                frag[3] = 'Dead Coral'


def make_statuses(tag, frag_ids, frags):

    # Fix consistency issues in detected type. Makes a double list of monitoring dates and estimated status of the fragment.
    # Insert result in the Status table.

    frag_index = 0
    frag_dates = []

    all_dates = get_monitoring_dates(tag)

    for f in frags:

        frag_date = {}

        frag_id = frag_ids[frag_index]
        frag_index += 1

        os = f[-1]
        obss = {}
        dates = deepcopy(all_dates)

        os.sort(key = lambda o: o['Date'])

        for o in os:
            obss[o['Date']] = {'Type': o['Type'], 'ObsId': o['ObservationId']}

        for date in dates:
            if date not in obss:
                obss[date] = {'Type': 'Undetected', 'ObsId': "NULL"}

        # Remove preceding 'undetected'

        i = 0

        while obss[dates[i]]['Type'] == 'Undetected':
            i += 1

        dates = dates[i:]

        status = []
        for d in dates:
            if(obss[d]['Type'] in ('Pocillopora', 'Acropora')):
                status.append('Live Coral')
            else:
                status.append(obss[d]['Type'])

        # Set all later statuses to Fallen for all end 'undetected'

        j = 1
        while status[-j] == 'Undetected':
            j += 1

        if j > 2:
            for k in range(1, j):
                status[-k] = 'Fallen'

        # Overwrite all other 'undetected' with next observation

        for l in range(len(status) - j, -1, -1):
            if status[l] == 'Undetected':
                status[l] = status[l + 1]

        # Set all later statuses to Dead if two consecutives Dead

        for i in range(1, len(status)):
            if status[i - 1] == 'Dead Coral':
                if status[i] == 'Dead Coral':
                    for j in range(i+1, len(dates)):
                        if status[j] in ['Live Coral', 'Bleached Coral']:
                            status[j] = 'Dead Coral'
                    break
                elif status[i] in ['Live Coral', 'Bleached Coral']:
                    status[i - 1] = status[i]

        # Get Falling/Dying date

        date0 = datetime.strptime('20' + str(dates[0]).zfill(6), '%Y%m%d').date()
        frag_date['Transplanted'] = date0.strftime('%Y-%m-%d')

        if f[3] in {'Acropora', 'Pocillopora'}:
            k = 0
            while k < len(status) - 1:
                if status[k + 1] in ['Fallen', 'Dead Coral']:
                    date1 = datetime.strptime('20' + str(dates[k]).zfill(6), '%Y%m%d').date()
                    date2 = datetime.strptime('20' + str(dates[k + 1]).zfill(6), '%Y%m%d').date()
                    frag_date['Dead'] = (date1 + (date2 - date1) / 2).strftime('%Y-%m-%d')
                    k = len(status)
                k += 1
        elif f[3] == 'Dead Coral':
            frag_date['Dead'] = date0.strftime('%Y-%m-%d')

        # Upload

        obs_ids =  [obss[d]['ObsId'] for d in dates]
        

        for i in range(0, len(status)):
            datem = datetime.strptime('20' + str(dates[i]).zfill(6), '%Y%m%d').date()
            insert_status(frag_id, obs_ids[i], status[i], dates[i], (datem - date0).days)

        frag_dates.append(frag_date)

    return(frag_dates)