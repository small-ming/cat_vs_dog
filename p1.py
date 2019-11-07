# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2

def rebuild(dir):
    for root,dirs,files in os.walk(dir):
        for file in files:
            filepath=os.path.join(root,file)
            try:
                image=cv2.imread(filepath)
                dim=(227,227)
                resized=cv2.resize(image,dim)
                path="cat_and_dog\cat\\"+file
                cv2.imwrite(path,resized)
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey(0)

rebuild('cat_and_dog\cat')