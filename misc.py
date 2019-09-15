# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:48:27 2019

@author: manda
"""
import os
import cv2
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import col_mom
def break_window(image_list):
    totColMom=[]
    #img = cv2.imread('C:/Users/manda/OneDrive/Desktop/ASU MWDB/Hand_0000002.jpg')
    
    for img in image_list:
        Y_tot_mean=[]
        U_tot_mean=[]
        V_tot_mean=[]
        Y_tot_dev=[]
        U_tot_dev=[]
        V_tot_dev=[]
        Y_tot_skew=[]
        U_tot_skew=[]
        V_tot_skew=[]
        Col_mom=[]
        img_yuv= cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        no_rows=100
        no_cols=100
        for i in range(0, img.shape[0], no_rows):
            for j in range(0, img.shape[1], no_cols):
                
                win=img_yuv[i:i+no_rows, j:j+no_cols]
                y,u,v=cv2.split(win)
                y_mean=np.mean(y)
                Y_tot_mean.append(y_mean)
                u_mean=np.mean(u)
                U_tot_mean.append(u_mean)
                v_mean=np.mean(v)
                V_tot_mean.append(v_mean)
                
                y_dev=np.std(y)
                Y_tot_dev.append(y_dev)
                u_dev=np.std(u)
                U_tot_dev.append(u_dev)
                v_dev=np.std(v)
                V_tot_dev.append(v_dev)
                
                y_array=np.array(y)
                y_flatten=y_array.flatten()
                y_skew=skew(y_flatten)
                Y_tot_skew.append(y_skew)
                
                u_array=np.array(u)
                u_flatten=u_array.flatten()
                u_skew=skew(u_flatten)
                U_tot_skew.append(u_skew)
                
                v_array=np.array(v)
                v_flatten=v_array.flatten()
                v_skew=skew(v_flatten)
                V_tot_skew.append(v_skew)
                
        Col_mom=Y_tot_mean+ U_tot_mean+ V_tot_mean+ Y_tot_dev+ U_tot_dev+ V_tot_dev+ Y_tot_skew+ U_tot_skew+V_tot_skew
        #col_mom.write_new_file(Col_mom)
        totColMom = totColMom + Col_mom
    return totColMom

def con_yuv(img):
     img = cv2.imread('C:/Users/manda/OneDrive/Desktop/ASU MWDB/Hand_0000002.jpg')
     img_yuv= cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
     return img_yuv
         
def con_gray(img):
     img = cv2.imread('C:/Users/manda/OneDrive/Desktop/ASU MWDB/Hand_0000002.jpg')
     gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     return gray
     
def write_new_file(fd):    
    with open('hog_res.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(fd) 
    
def plot_image(img):
    plt.imshow(img, cmap='Greys_r')
    plt.show()
    
def get_images_in_directory(path):
    dirname = os.path.dirname(__file__)
    print(dirname)
    complete_path = os.path.join(dirname, path)
    print("Complete path", complete_path)
    files = {}
    for filename in os.listdir(complete_path):
        # print("File", filename)
        files[filename] = os.path.join(complete_path, filename)
    return files


def read_image(image_path, gray=False):
    # print(os.getcwd())
    dirname = os.path.dirname(__file__)
    image = mpimg.imread(os.path.join(dirname, image_path))
    if gray:
        image = con_gray(image)
    return image