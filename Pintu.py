import tensorflow as tf
import cv2
# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import os
from segmentation import get_roi
from PIL import Image
import tifffile as tiff
import math

# It is create for UNet-in-Tensorflow

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

saver = tf.train.import_meta_graph("./baseline/model.ckpt.meta")
sess = tf.InteractiveSession()
saver.restore(sess, "baseline/model.ckpt")
X, mode = tf.get_collection("inputs")
pred = tf.get_collection("outputs")[0]

def make_dir(path):

    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)



def get_roi(image, start_i, start_j, size, stride, channel):

    if channel == 1:
        return image[start_i * stride:start_i * stride + size, start_j * stride : start_j * stride + size]
    else:
        return image[start_i * stride:start_i * stride + size, start_j * stride : start_j * stride + size, :channel]


def read_image(image_path, gray=False):
    """Returns an image array

    Args:
        image_path (str): Path to image.jpg

    Returns:
        3-D array: RGB numpy image array
    """
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pipeline(image, threshold=0.9999, image_WH=(256, 256)):
    image = np.copy(image)
    H, W ,C = image.shape

    if (W, H) != image_WH:
        image = cv2.resize(image, image_WH)

    mask_pred = sess.run(pred, feed_dict={X: np.expand_dims(image, axis = 0),
                                          mode: False})
    mask_pred = np.squeeze(mask_pred)
    mask_pred = mask_pred > threshold

    return(mask_pred)


def minor_merge(img,save_path,save_path2,size):

    # padding image
    img = tiff.imread(img)
    s1= int(math.floor(img.shape[0]/256))
    s2 = int(math.floor(img.shape[1]/256))

    imgarr = img[:s1*256, :s2*256, :]
    gap = (512 - size) / 2
    shape_0 = math.ceil(1.0 * imgarr.shape[0] / size) * size + 2 * gap
    shape_0 = int(shape_0)
    shape_1 = math.ceil(1.0 * imgarr.shape[1] / size) * size + 2 * gap
    shape_1 = int(shape_1)
    padding = ((gap, shape_0 - gap - imgarr.shape[0]), (gap, shape_1 - gap - imgarr.shape[1]), (0, 0))
    Image_pad = np.pad(imgarr, padding, 'reflect')
    # cv2.imwrite('merge1.png', Image_pad)
    # count=0
    # for i in range(imgarr.shape[0] / 512):
    #     for j in range(imgarr.shape[1] / 512):
    #         testslice = get_roi(imgarr, i, j, 512, 512, 3)
    #         imgname = save_path + str(count) + ".jpg"
    #         cv2.imwrite(imgname, testslice)
    #         print('(Random is false)image_name:', imgname)
    #         count += 1

    count = 0
    for i in range(shape_0 / size):
        for j in range(shape_1 / size):
            imgslice = get_roi(Image_pad, i, j, size=256, stride=size, channel=3)
            # labslice = labarr[i * 512:i * 512 + 512, j * 512:j * 512 + 512]
            imgname = save_path + '/' + str(count) + ".jpg"
            # labname = "./data/testlabel2/" + str(count) + ".jpg"
            misc.imsave(imgname, imgslice)
            # misc.imsave(labname, labslice)
            print('image_name:', imgname)
            count += 1

    imglist = os.listdir(save_path)
    imglist.sort(key=lambda x:int(x.split('.')[0]))
    print(imglist)
    # col = shape_1/size
    for i in imglist:
        img = Image.open(save_path+'/'+i)
        img = np.array(img)
        image_pred = pipeline(img)
        image_pred = image_pred.astype(int) * 255
        image_pred = np.expand_dims(image_pred, axis=2)
        print('testing image:',i)
        merge_img = np.zeros((size, size*2, 3), dtype=int)

        merge_img[:, :size, :] = img[gap:gap+size,gap:gap+size,:]
        image_pred = np.concatenate([image_pred, image_pred, image_pred], axis=2)
        merge_img[:, size:2*size, :] = image_pred[gap:gap+size,gap:gap+size,:]
        misc.imsave(save_path2 + i, merge_img)


def minor_Pintu(img,save_path,size):
    """
    :param img:
    :param savepath:
    :param merge_size:
    :return:
    """
    # padding image

    imgarr = tiff.imread(img)
    gap = (256 - size) / 2
    shape_0 = math.ceil(1.0 * imgarr.shape[0] / size) * size + 2 * gap
    shape_0 = int(shape_0)
    shape_1 = math.ceil(1.0 * imgarr.shape[1] / size) * size + 2 * gap
    shape_1 = int(shape_1)
    pading = ((gap, shape_0 - gap - imgarr.shape[0]), (gap, shape_1 - gap - imgarr.shape[1]), (0, 0))
    Image_pad = np.pad(imgarr, pading, 'reflect')
    # cv2.imwrite('merge1.png', Image_pad)
    count = 0
    for i in range(shape_0 / size):
        for j in range(shape_1 / size):
            imgslice = get_roi(Image_pad, i, j, size=256, stride=size, channel=3)
            # labslice = labarr[i * 512:i * 512 + 512, j * 512:j * 512 + 512]
            imgname = save_path + '/' + str(count) + ".jpg"
            # labname = "./data/testlabel2/" + str(count) + ".jpg"
            misc.imsave(imgname, imgslice)
            # misc.imsave(labname, labslice)
            print('image_name:', imgname)
            count += 1
    imglist = os.listdir(save_path)
    imglist.sort(key=lambda x:int(x.split('.')[0]))
    print(imglist)
    toImage = np.zeros([shape_0-2*gap,shape_1-2*gap])
    col = shape_1/size
    for i in imglist:
        img = Image.open(save_path+'/'+i)
        image_pred = pipeline(img)
        img = image_pred.astype(int) * 255
        img = img[gap:gap+size,gap:gap+size]
        # image_pred = np.concatenate(([img],[img],[img]),axis=2)
        # image_pred = Image.fromarray(image_pred)
        print('testing image:',i)
        loc = (((int(i.split('.')[0])//col)*size),((int(i.split('.')[0])%col)*size))
        print(loc)
        toImage[loc[0]:loc[0]+size,loc[1]:loc[1]+size]=img
    toImage = toImage[:imgarr.shape[0], :imgarr.shape[1]]
    misc.imsave('2011hist_pred.tif',toImage)

def compare(img_path1,img_path2):
    """
    Merging two images to compare results.
    :param img_path1:
    :param img_path2:
    :return:
    """
    imglist_1=os.listdir(img_path1)
    imglist_1.sort(key=lambda x: int(x.split('.')[0]))
    for i in imglist_1:
        img_1 = Image.open(img_path1+'/'+i)
        img_2 = Image.open(img_path2+'/'+i)
        merge = np.concatenate((img_1,img_2),axis=0)
        misc.imsave('./data/compare/'+i,merge)

if __name__ == '__main__':
    img = './2011hist_match.tif'
    print(img)
    make_dir('./data/pred')
    # make_dir('./data/pred_hist')
    # minor_merge(img,'./data/pred','./data/pred_hist/',400)
    minor_Pintu(img,'./data/pred',200)
    # make_dir('./data/compare')
    # compare('./data/pred_bright','./data/pred_hist')