# ---------------------------------------------------------------------
# Exercises from lesson 2 (object detection)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import open3d as o3d
import math
import numpy as np
import zlib

import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt


# Exercise C2-4-6 : Plotting the precision-recall curve
def plot_precision_recall(recall, precision): 

    # Please note: this function assumes that you have pre-computed the precions/recall value pairs from the test sequence
    #              by subsequently setting the variable configs.conf_thresh to the values 0.1 ... 0.9 and noted down the results.
    
    # Please create a 2d scatter plot of all precision/recall pairs 
    plt.scatter(recall, precision,)
    plt.show()



# Exercise C2-3-4 : Compute precision and recall
def compute_precision_recall(det_performance_all, conf_thresh=0.5):

    if len(det_performance_all)==0 :
        print("no detections for conf_thresh = " + str(conf_thresh))
        return
    
    # extract the total number of positives, true positives, false negatives and false positives
    # format of det_performance_all is [ious, center_devs, pos_negs]
    # [TP, FP, TN, FN]
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for det_performance in det_performance_all:
        #print(det_performance[2])
        true_positives += det_performance[2][1]
        false_negatives += det_performance[2][2]
        false_positives += det_performance[2][3]

    # print("TP = " + str(true_positives) + ", FP = " + str(false_positives) + ", FN = " + str(false_negatives))
    
    # compute precision
    precision = true_positives/ (true_positives + false_positives)
    
    # compute recall 
    recall = true_positives/ (true_positives + false_negatives)

    print("precision = " + str(precision) + ", recall = " + str(recall) + ", conf_thres = " + str(conf_thresh) + "\n")    
    return precision, recall
    



# Exercise C2-3-2 : Transform metric point coordinates to BEV space
def pcl_to_bev(lidar_pcl, configs, vis=True):

    # compute bev-map discretization by dividing x-range by the bev-image height
    x_range = configs.lim_x[1] -configs.lim_x[0]
    bev_map_disc_x = x_range / configs.bev_height

    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_copy = np.copy(lidar_pcl)
    lidar_pcl_copy[:, 0] = np.int_(np.floor(lidar_pcl_copy[:,0] / bev_map_disc_x)) 
    
    # transform all metrix y-coordinates as well but center the foward-facing x-axis on the middle of the image
    y_range = configs.lim_y[1] - configs.lim_y[0]
    bev_map_disc_y = y_range / configs.bev_width
    y_offset = (bev_map_disc_y +1)/2
    lidar_pcl_copy[:, 1] = np.int_(np.floor(lidar_pcl_copy[:,1] / bev_map_disc_y) + y_offset) 
 
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl_copy[:,2] = lidar_pcl_copy[:,2] - configs.lim_z[0]

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    index_sort = np.lexsort(( -lidar_pcl_copy[:,2], lidar_pcl_copy[:,1], lidar_pcl_copy[:,0]))
    lidar_pcl_height = lidar_pcl_copy[index_sort]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    _, index_unique = np.unique(lidar_pcl_height[:, 0:2], axis=0, return_index=True)
    lidar_pcl_height = lidar_pcl_height[index_unique]

    # assign the height value of each unique entry in lidar_top_pcl to the height map and 
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[np.int_(lidar_pcl_height[:,0]), np.int_(lidar_pcl_height[:,1])] = lidar_pcl_height[:,2] / float(np.abs(configs.lim_z[1]- configs.lim_z[0]))
    
    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
    lidar_pcl_copy[lidar_pcl_copy[:,3]>1.0,3] = 1.0
    index_intensity = np.lexsort((lidar_pcl_copy[:, 3], lidar_pcl_copy[:,1], lidar_pcl_copy[:,0]))
    lidar_pcl_intensity = lidar_pcl_copy[index_intensity]

    # only keep one point per grid cell
    _, index_unique = np.unique(lidar_pcl_intensity[:, 0:2], axis=0, return_index=True)
    lidar_pcl_intensity = lidar_pcl_intensity[index_unique]
    print(lidar_pcl_intensity)
    
    # create the intensity map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_intensity[:,0]), np.int_(lidar_pcl_intensity[:,1])] = lidar_pcl_intensity[:,3] / (np.amax(lidar_pcl_intensity[:,3]- np.amin(lidar_pcl_intensity[:,3])))

    # visualize intensity map
    if vis:
        img_intensity = intensity_map * 256
        img_intensity = img_intensity.astype(np.uint8)
        while (1):
            cv2.imshow('img_intensity', img_intensity)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    
def test_vis(image):

    # convert the actual image into rgb format
    img = np.array(image, dtype=np.uint8)

    # resize the image to better fit the screen
    dim = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    resized = cv2.resize(img, dim)

    # display the image 
    cv2.imshow("liader vis", resized)
    cv2.waitKey(0)