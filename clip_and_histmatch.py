import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from scipy import misc

"""
label与原图相乘，拼图
"""
path1 = './final/image_or/image/'
path2 =  './final/image_or/people/'
path3 = './final/image_or/yuanpan/'
path_result = './result/'
img_list = os.listdir(path1)
img_list.remove(img_list[0])
label_list = os.listdir(path2)       # body label
label_list.remove(label_list[0])
label_list2 = os.listdir(path3)      # disk label
label_list2.remove(label_list2[0])
count = 0
for i in img_list:
    img = mpimg.imread(path1+i)      #body
    label = mpimg.imread(path2+i)     #body label
    label = np.expand_dims(label, axis=-1)
    label = np.concatenate([label,label,label],axis=2)

    label2 = mpimg.imread(path3+i)     #disk label
    label2.flags.writeable = True
    label2 = label2/255
    # label2[:,:,1] = 0
    # label2[:,:,2] = 0
    # label2 = np.expand_dims(label2, axis=-1)
    # label2 = np.concatenate([label2,label2,label2],axis=2)
    rst= img*label2 +(1-label2)*label
    merge_image = np.zeros([img.shape[0],img.shape[1]*2,3])
    merge_image[:,:img.shape[1],:] = img
    merge_image[:,img.shape[1]:img.shape[1]*2,:] = rst
    misc.imsave(path_result+i,merge_image)