from __future__ import print_function
import cv2
import imutils
import numpy as np
import os
current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'train1')
n=0
for f in os.listdir(image_dir):
    img = cv2.imread("train1/"+f)
    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    original = img
    gamma = 1.5
    adjusted = adjust_gamma(original, gamma=gamma)
    rotated1 = imutils.rotate_bound(img, 2)
    rotated2 = imutils.rotate_bound(img, -2)
    rotated3 = imutils.rotate_bound(img, 4)
    rotated4 = imutils.rotate_bound(img, -4)
    rotated5 = imutils.rotate_bound(img, 6)
    rotated6 = imutils.rotate_bound(img, -6)
    rotated7 = imutils.rotate_bound(img, 8)
    rotated8 = imutils.rotate_bound(img, -8)
    cv2.imwrite("train2/"+f+".jpg",img)
    cv2.imwrite("train2/"+f+"_bright.jpg",adjusted)
    cv2.imwrite("train2/"+f+"_rot1.jpg",rotated1)    
    cv2.imwrite("train2/"+f+"_rot2.jpg",rotated2)    
    cv2.imwrite("train2/"+f+"_rot3.jpg",rotated3)    
    cv2.imwrite("train2/"+f+"_rot4.jpg",rotated4)    
    cv2.imwrite("train2/"+f+"_rot5.jpg",rotated5)    
    cv2.imwrite("train2/"+f+"_rot6.jpg",rotated6)    
    cv2.imwrite("train2/"+f+"_rot7.jpg",rotated7)    
    cv2.imwrite("train2/"+f+"_rot8.jpg",rotated8)    

    print(f)
