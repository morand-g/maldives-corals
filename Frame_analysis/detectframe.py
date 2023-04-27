import os

from math import ceil
import numpy as np
from mpi4py import MPI

import random

import cv2
from skimage.util.dtype import img_as_ubyte, img_as_float
from skimage.filters import frangi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from stochopy import Evolutionary

from sqlfunctions import insert_frameparams

#####################################################################################
#                                                                                   #
# This file contains all the needed functions to detect the frame bars in a picture #
#               A description is included in each function definition               #
#                                                                                   #
#####################################################################################

def original(image_path, scale = 1):

    # Returns original picture at the specified scale

    img = cv2.imread(image_path)
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return(cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA))


def cloud_transform(focal = -150, rotation = [np.pi / 6, 0, 0], translation = [0, 150, 0], view = "H00", display = False, points = []):

    # Transforms 3d points into photographic projection (which still needs to be scaled and translated to fit the picture coordinates)
    # If no points are specified, uses the bars we expect to see with the specified view.

    if len(points) == 0:
        firstpoints = np.loadtxt('/home/mdc/Desktop/Frame analysis/SmallFrameLines.csv', skiprows = 1, delimiter = ',', usecols = [0, 1, 2])
        secondpoints = np.loadtxt('/home/mdc/Desktop/Frame analysis/SmallFrameLines.csv', skiprows = 1, delimiter = ',', usecols = [3, 4, 5])

        if view == "H00" or view == "H06":
            points = np.vstack([firstpoints[:19], np.flipud(secondpoints[:19])])
        elif view == "H03" or view == "H09":
            points = np.vstack([firstpoints[:23], np.flipud(secondpoints[:23])])

        # Include top bars only if we are visualizing the result

        if(not(display)) :
            points = points[6:-6]
    
    # Applies a rotation, then a translation to fit the camera position
    pitch = rotation[0]   # around x axis
    roll = rotation[1]    # around y axis
    yaw = rotation[2]     # around z axis

    Rx = np.array([[1, 0, 0, 0], [0, np.cos(pitch), -np.sin(pitch), 0], [0, np.sin(pitch), np.cos(pitch), 0], [0, 0, 0, 1]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll), 0], [0, 1, 0, 0], [-np.sin(roll), 0, np.cos(roll), 0], [0, 0, 0, 1]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = Rx @ (Ry @ Rz)
    
    T = np.array([[1, 0, 0, translation[0]], [0, 1, 0, translation[1]], [0, 0, 1, translation[2]], [0, 0, 0, 1]])

    M = T @ R

    original_points = np.transpose(np.array(points))
    original_points = np.vstack([original_points, np.ones(original_points.shape[1])])
    new_points = M @ original_points

    # Divide all coordinates by the distance to the camera (second coordinate) to compute homogenous coordinates.
    new_x = focal * new_points[0] / new_points[1]
    new_z = focal * new_points[2] / new_points[1]

    projected_points = np.vstack([new_x, new_points[1], new_z])
    
    # Returns a list of 3D points, x and z being the projected coordinates and y the distance to the camera.
    return(np.transpose(projected_points))


def score(view, mask, focal = -150, Rx = np.pi / 6, Ry = 0, Rz = 0, scale = 0.01, Tu = 0, Tv = 0, debug = False):
    
    # Returns a measurement of the fit between the specified projection and the given mask. Lower is better.

    h, w = mask.shape
    
    projection = np.zeros((h, w))

    # Camera projection
    proj_line_points = cloud_transform(focal, [Rx, Ry, Rz], view=view)

    scaling = w * scale

    radius = 0.75 * scaling # 1.5
    Timage = [w * Tu, h * Tv]

    # Scale and translate points to fit the picture plane, and then draw projected bars.
    for i in range(len(proj_line_points) // 2):

        x1, y1 = -scaling * proj_line_points[i][0] + Timage[0], scaling * proj_line_points[i][2] + Timage[1]
        x2, y2 = -scaling * proj_line_points[-i-1][0] + Timage[0], scaling * proj_line_points[-i-1][2] + Timage[1]
        cv2.line(projection, (int(x1), int(y1)), (int(x2), int(y2)), 1, ceil(radius * 2))
    
    ## Saves 10% of projected pictures for debug
    if(debug):
        cv2.imwrite('/tmp/projection' + str(random.random()) + '.jpg', projection)
    
    # Number of matching frame pixels :
    tp = np.sum(mask * projection) + 1

    # Mixture of false positives and false negatives. Values optimized empirically after numerous tests!
    false = (17 * np.sum((mask - projection) == 1) + 3 * np.sum((projection - mask) == 1))
    
    return(false / tp)


def optim_from_mask(mask, view, mask_w, different_angle = False, pillar = None):

    # Runs the projection process, with the given mask

    if view == "H00" or view == "H06":
        angle = 0
        if different_angle:
            if pillar == 'right':
                angle = np.pi / 6
            else:
                angle = -np.pi / 6

    elif view == "H03" or view == "H09":
        angle = - np.pi / 6
        if different_angle:
            if pillar == 'right':
                angle = 0
            else:
                angle = - np.pi / 3
    else:
        view = 'H00'
        angle = 0
    
    # Variables to optimize : focal, Rx, Ry, Rz, scaling, Tu, Tv
    # bnds = [(-180, -70), (np.pi / 12, np.pi / 3), (-np.pi/18, np.pi/18), (angle - np.pi / 12, angle + np.pi / 12), (0.006, 0.012), (0.35, 0.65), (0.3, 0.8)]
    bnds = [(-220, -70), (np.pi / 12, np.pi / 3), (-np.pi/12, np.pi/12), (angle - np.pi / 12, angle + np.pi / 12), (0.006, 0.012), (0.3, 0.6), (0.3, 0.8)]
    
    lower = np.array([i[0] for i in bnds])
    upper = np.array([i[1] for i in bnds])

    # ea = Evolutionary(lambda x : score(view, mask, *x), lower = lower, upper = upper, max_iter = 3000, popsize = 10, constrain = True, random_state = -1)
    ea = Evolutionary(lambda x : score(view, mask, *x), lower = lower, upper = upper, max_iter = 5000, popsize = 10, constrain = True, random_state = -1)
    xopt, gfit = ea.optimize(solver = 'de', F = 0.7, CR = 0.2)#, xstart = expectedx, sigma = 0.8)

    sc = score(view, mask, *xopt, debug = False)

    # Returns the best projection parameters, and the best score

    return(xopt, gfit)


def save_projected_picture(root_folder, image_name, params):

    # Save original picture with frame projection superimposed

    image_path = root_folder + 'Originals/' + image_name
    background = original(image_path)
    h, w, _ = background.shape
    view = image_name[6:9] 

    if view[0] != 'H':
        view = 'H00'
        
    final_points = cloud_transform(params[0], [params[1], params[2], params[3]], view = view, display = True)

    Timage = [w  * params[5], h * params[6]]
    scaling = params[4] * w
    radius = -0.005 * scaling * params[0]

    for i in range(len(final_points) // 2):

        x1, y1 = -scaling * final_points[i][0] + Timage[0], scaling * final_points[i][2] + Timage[1]
        x2, y2 = -scaling * final_points[-i-1][0] + Timage[0], scaling * final_points[-i-1][2] + Timage[1]
        cv2.line(background, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), ceil(radius * 2))

    cv2.imwrite(root_folder + 'Frames/' + image_name, background)


def compute_camera_params(root_folder, unet_masks, metadata, save_pics = True):

    # Run the optimization and try different options if needed (removing background, different angle)
    # Runs parallel threads on all CPU cores

    j = 0

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    for i in range(len(metadata)):
        filename = metadata[i][0]

        if i%size!=rank: continue

        unet_mask = unet_masks[i]

        h, w = unet_mask.shape

        view = filename[6:9]
        
        params, sc = optim_from_mask(unet_mask, view, w)

        if sc > 18:
            filtered = remove_background_frames(unet_mask)
            cv2.imwrite(root_folder + 'Masks/' + filename[:-4] + 'F.png', 255 * filtered)
            params2, sc2 = optim_from_mask(filtered, view, w)

            if sc2 < sc:
                params, sc = params2, sc2

        if sc > 18:
            params3, sc3 = optim_from_mask(unet_mask, view, w, True, pillarside(root_folder, filename))

            if sc3 < sc:
                params, sc = params3, sc3

        if sc > 18:
            filtered2 = remove_background_frames(unet_mask, intensity = 2)
            cv2.imwrite(root_folder + 'Masks/' + filename[:-4] + 'F.png', 255 * filtered2)
            params4, sc4 = optim_from_mask(filtered2, view, w)

            if sc4 < sc:
                params, sc = params4, sc4

        if(save_pics):
            save_projected_picture(root_folder, filename, params)

        insert_frameparams(filename, sc, params)

    print("Core {}: Done".format(rank))


def remove_background_frames(mask, intensity = 1):

    # Filters the mask keeping only the portion where the frame is expected to be.

    h, w = mask.shape[:2]

    if intensity == 1:
        filt = cv2.imread('mask_filter.png', 0)
    elif intensity == 2:
        filt = cv2.imread('mask_filter2.png', 0)
    adapt_filter = cv2.resize(filt, (w, h), interpolation = cv2.INTER_AREA)

    return(mask*adapt_filter)


def pillarside(root_folder, filename):

    # Returns the darkest side of the monitoring picture, to detect pillars / dark obstacles responsible for angle offset.

    image_path = root_folder + 'Originals/' + filename
    pic = original(image_path)

    h, w = pic.shape[:2]

    margin = int(0.05 * w)
    left_average = np.sum(pic[:,:margin])
    right_average = np.sum(pic[:,-margin:])

    if left_average < right_average:
        return('left')
    else:
        return('right')


############################################################################################################################
#                                                                                                                          #
#                                                   OLD functions                                                          #
#                                                 Not in use anymore                                                       #
#                                                                                                                          #
############################################################################################################################


def preprocess(image_path, detector = 'Ridge', cliplimit = 1, grid = 1, sig = 1.5, target_size = 400):

    # Applies preprocessing transformations to original picture, at the specified scale.

    ##### NOT USED IN DEEP LEARNING METHOD #####

    # Resize
    img = original(image_path)
    w = img.shape[1]
    scale = target_size / w
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    
    # Apply CLAHE on the blue component
    blue = img[:,:,1]
    clahe = cv2.createCLAHE(cliplimit, tileGridSize=(grid, grid))
    cl_b = clahe.apply(blue)

    # Apply specified filter
    if detector == 'Ridge':
        H_elems = hessian_matrix(img_as_float(cl_b), sigma=sig, order='rc')
        c, _ = hessian_matrix_eigvals(H_elems)
        m = np.max(np.abs(c))

    elif detector == 'Frangi':
        c = frangi(img_as_float(cl_b))
        m = np.max(np.abs(c))

    # Apply CLAHE again
    image = img_as_ubyte(c / (m + 0.01))
    final = clahe.apply(image)

    return(final)


def read_points(image_path, scale = 1, margin = 0.15):

    # Returns line segments detected by LSD algorithm. Filters lines in the margins and short lines.

    ##### NOT USED IN DEEP LEARNING METHOD #####

    ## line arguments
    # 0 : xA
    # 1 : yA
    # 2 : xB
    # 3 : yB
    # 4 : a
    # 5 : b
    # 6 : theta
    # 7 : bintheta
    # 8 : width
    
    image_name = os.path.basename(image_path)
    selected_lines = []
    
    with open('/tmp/' + image_name[:-4] + '.txt', 'r') as f:
        lines = f.readlines()

        img = original(image_path, scale)
        im_height, im_width, _ = img.shape

        for line in lines:
            data = line.split(' ')
            x1 = float(data[0])
            y1 = float(data[1])
            x2 = float(data[2])
            y2 = float(data[3])
            width = float(data[4])
            if(x1 == x2):
                x2 = x2 + 1
            a = (y2 - y1) / (x2 - x1)
            b = y1 - x1 * a
            theta = np.arctan(a)
            bintheta = np.floor(np.divide(theta, np.pi/180))

            marginw = im_width * margin
            marginh = im_height * margin
            borderTest = (max(x1,x2) < marginw) or (max(y1,y2) < marginh) or (min(x1,x2) > im_width - marginw) or (min(y1,y2) > im_height - marginh) 
            length = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )

            if(length > 50 * scale and not(borderTest)):
                selected_lines.append([x1, y1, x2, y2, a, b, theta, bintheta, width])
        
        return(np.array(selected_lines))


def cluster(segments, t, tol):

    # Clusters all segments based on  specified parameter (theta, b, or longitudinal gap), with specified tolerance.

    ##### NOT USED IN DEEP LEARNING METHOD #####

    toCluster = np.asarray(segments)
    
    if(len(toCluster) <= 1) :

        return([toCluster])

    else :

        # Define metric used to compute distances

        if(t == 'theta') :

            method = 'centroid'

            def dist(p1, p2):

                d_theta = (abs(p1[6] - p2[6]))
                return(min(np.pi - d_theta, d_theta))

        elif(t == 'b') :

            method = 'centroid'

            def dist(p1, p2):

                dy = np.tan(p1[6]) / np.sqrt(1 + np.tan(p1[6])**2) + np.tan(p2[6]) / np.sqrt(1 + np.tan(p2[6])**2)
                dx = 1 / np.sqrt(1 + np.tan(p1[6])**2) + 1 / np.sqrt(1 + np.tan(p2[6])**2)
                a = dy / dx
                theta = np.arctan(a)
                x1, y1 = (p1[0] + p1[2]) / 2, (p1[1] + p1[3]) / 2
                x2, y2 = (p2[0] + p2[2]) / 2, (p2[1] + p2[3]) / 2
                b1 = y1 - a * x1
                b2 = y2 - a * x2
                d_b = abs(b1 - b2) * np.cos(theta)
                return(d_b)

        elif(t == 'gap') :

            method = 'single'

            def dist(p1, p2):

                dy = np.tan(p1[6]) / np.sqrt(1 + np.tan(p1[6])**2) + np.tan(p2[6]) / np.sqrt(1 + np.tan(p2[6])**2)
                dx = 1 / np.sqrt(1 + np.tan(p1[6])**2) + 1 / np.sqrt(1 + np.tan(p2[6])**2)
                a = dy / dx
                theta = np.arctan(a)

                Slen1 = np.sqrt((p1[0] - p1[2])**2 + (p1[1] - p1[3])**2)
                Slen2 = np.sqrt((p2[0] - p2[2])**2 + (p2[1] - p2[3])**2)
                x1, y1 = (p1[0] + p1[2]) / 2, (p1[1] + p1[3]) / 2
                x2, y2 = (p2[0] + p2[2]) / 2, (p2[1] + p2[3]) / 2
                b1, b2 = y1 - a * x1,  y2 - a * x2
                b = 0.5 * (b1 + b2)

                xA1, xB1 = x1 - np.cos(theta) * Slen1 / 2, x1 + np.cos(theta) * Slen1 / 2
                xA2, xB2 = x2 - np.cos(theta) * Slen2 / 2, x2 + np.cos(theta) * Slen2 / 2

                xSAP1, xSBP1 = xA1 + (b1 - b) * a / (1 + a**2), xB1 + (b1 - b) * a / (1 + a**2)
                xSAP2, xSBP2 = xA2 + (b2 - b) * a / (1 + a**2), xB2 + (b2 - b) * a / (1 + a**2)

                if (xSAP1 - xSBP2) * (xSAP2 - xSBP1) >= 0:
                    return(0)
                else:
                    return(min(abs(xSAP1 - xSBP2), abs(xSBP1 - xSAP2)) / np.cos(theta))


        dm = pdist(toCluster, dist)
        dendogram = linkage(dm, method)
        clusters = np.asarray([fcluster(dendogram, tol, criterion = 'distance')]).T

        segments = np.hstack([toCluster, clusters])
                
        xrange = range(0, np.max(clusters))
        master_list = [[] for x in xrange]
        
        for s in segments:
            master_list[int(s[-1]) - 1].append(list(s[:-1]))

        # Returns a list of clusters, each cluster being a list of segments.

        return(master_list)


def merge(segments):

    # Merges all specified segments into an 'average' line

    ##### NOT USED IN DEEP LEARNING METHOD #####

    segments= np.asarray(segments)

    if len(segments) <= 1 :
        return(segments)

    else :
        Slen = np.sqrt((segments[:,0] - segments[:,2])**2 + (segments[:,1] - segments[:,3])**2) 
       
        dy = np.sum(Slen * np.tan(segments[:,6]) / np.sqrt(1 + np.tan(segments[:,6])**2))
        dx = np.sum(Slen / np.sqrt(1 + np.tan(segments[:,6])**2))
        a = dy / dx
        theta = np.arctan(a)
        xS, yS = (segments[:,0] + segments[:,2]) / 2, (segments[:,1] + segments[:,3]) / 2
        bS = yS - a * xS
        b = np.sum(Slen * bS) / np.sum(Slen)

        xSA, xSB = xS - np.cos(theta) * Slen / 2, xS + np.cos(theta) * Slen / 2

        def projX(x): return(x + (bS - b) * a / (1 + a**2))

        d = np.sqrt((a * (xSA - projX(xSA)) + bS - b)**2 + (xSA - projX(xSA))**2)

        xA, xB = min(np.hstack([projX(xSA), projX(xSB)])), max(np.hstack([projX(xSA), projX(xSB)]))
        yA, yB = a * xA + b, a * xB + b

        
        width = np.average(np.array(d) + segments[:,8])

        line = [xA, yA, xB, yB, a, b, theta, 0, width]

        return([line])


def analytical_mask(image_path, processing_size = 400, mask_size = (400, 300), horizontal_offset = 0):

    ##### Compute a mask by applying preprocessing, LSD and segment merging

    image_name = os.path.basename(image_path)
    view = image_name[6:9]
    img = original(image_path)
    w = img.shape[1]
    processing_scale = processing_size / w
    mask_scale = mask_size[0] / w
    mask_scaling = mask_scale / processing_scale

    ###################   Preprocess and apply LSD ########################

    cv2.imwrite('/tmp/' + image_name, preprocess(image_path, target_size = processing_size))
    os.system('convert /tmp/' + image_name + ' -flatten /tmp/' + image_name[:-4] + '.pgm')
    os.system('./lsd /tmp/' + image_name[:-4] + '.pgm /tmp/' + image_name[:-4] + '.txt')

    ################ Create picture with LSD segments #####################

    lines = read_points(image_path, processing_scale)
    background = original(image_path, mask_scale)

    for line in lines:
        xA, yA = mask_scaling * line[0], mask_scaling * line[1]
        xB, yB = mask_scaling * line[2], mask_scaling * line[3]
        cv2.line(background, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), ceil(mask_scaling * line[8]))

    cv2.imwrite('/tmp/' + image_name[:-4] + '_LSD.jpg', background)

    ########################### Line merge ################################

    L = np.array(read_points(image_path, processing_scale))
    merged_lines = []
    
    b_tol = 50 * processing_scale
    theta_tol = 0.15
    max_gap = 120 * processing_scale
    
    # Cluster segments according to theta, then according to b, then according to longitudinal gap, and merge resulting clusters

    theta_clusters = cluster(L, 'theta', theta_tol)

    for theta_cluster in theta_clusters :

        b_clusters = cluster(theta_cluster, 'b', b_tol)
        for b_cluster in b_clusters :

            gap_clusters = cluster(b_cluster, 'gap', max_gap)
            for gap_cluster in gap_clusters :

                l = merge(gap_cluster)
                merged_lines.extend(l)

    ################# Create picture with merged lines ####################
                
    background = original(image_path, mask_scale)
    h, w, _ = background.shape
    black = np.zeros((h, w))

    for line in merged_lines:
        
        xA, yA = mask_scaling * line[0], mask_scaling * line[1]
        xB, yB = mask_scaling * line[2], mask_scaling * line[3]
        width = mask_scaling * max(line[8], 1)
        l = np.sqrt((xB-xA)**2 + (yB-yA)**2)
        theta = line[6]

        if (l > 100 * mask_scale) :# and width > 20 * scale) :

            if abs(theta) <= np.pi / 6:
                yA -= horizontal_offset * mask_size[1]
                yB -= horizontal_offset * mask_size[1]

            cv2.line(background, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), ceil(width / 2))
            cv2.line(black, (int(xA), int(yA)), (int(xB), int(yB)), 1, ceil(width / 2))

    cv2.imwrite('/tmp/' + image_name[:-4] + '_merged.png', background)

    h_diff = mask_size[1] - black.shape[0]
    vert_border = (h_diff // 2, h_diff // 2 + h_diff % 2)
    padded = cv2.copyMakeBorder(black, vert_border[0], vert_border[1], 0, 0, cv2.BORDER_CONSTANT, 0)

    return(padded)
