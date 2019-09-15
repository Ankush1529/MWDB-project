# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:18:15 2019

@author: manda
"""
import cv2
import misc
import os

from skimage.feature import hog

#image = data.load(Hand_0007759.png,as_grey=False)
#image = mpimg.imread('C:/Users/manda/OneDrive/Desktop/ASU MWDB/Hand_0000002.jpg')
image_path='Hand_0000002.jpg'
img = misc.read_image(os.path.join(os.path.dirname(__file__), image_path))
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray =cv2.resize(gray, (120, 160))

fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2))
        
misc.write_new_file(fd)
