import colorsys
import glob
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib
import shutil
import re
import wget
import pandas
from matplotlib import pyplot as plt
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from numpy.random import rand
import sklearn
import pandas as pd
from random import randint
from itertools import groupby
import skimage
from colorir import sRGB
from PIL import Image, ImageDraw, ImageFont
from scipy.special import softmax
from array import *
from scipy.special import softmax
import csv
import pandas
from matplotlib import pyplot as plt
import cv2
import matplotlib.pyplot as plt


# with open('AppIconsCSV.csv') as file:
#     reader = csv.reader(file)     #читаем файл со всеми адресами картинок
#     my_list_url = list(map(tuple, reader))
# img_list = []
# for id, image_filename in enumerate(my_list_url):
#     image_numpy = skimage.io.imread(image_filename[0])
#     resized_image = cv2.resize(image_numpy, (64, 64))
#     img_list.append(resized_image)
#     im = Image.fromarray(resized_image)
#     # im.save("imgs17000" + os.sep + str(id).zfill(5) + ".jpeg")
#     if id==1000:
#         np.save('img_list', img_list)
#
#         break
#
# print('DONE!')

img_list = np.load('img_list.npy')
# all_pix = []
# for er, ut in enumerate(img_list):
#     img_p = np.concatenate(ut)
#     all_pix.append(img_p)
#     if er == len(img_list):
#         np.save('all_pix', all_pix)
#         break
# np.save('all_pix', all_pix)
all_pix = np.load('all_pix.npy')
#
clusters_n = 32
all = all_pix.reshape(all_pix.shape[0]*all_pix.shape[1], 3)
# np.save('all', all)
# all = np.load('all.npy')
kmeans_s = KMeans(n_clusters=int(clusters_n)).fit(all)
ClusterCentersPIC1 = np.array(kmeans_s.cluster_centers_, dtype="uint8")
np.save('ClusterCentersPIC1', ClusterCentersPIC1)
inp_img_list = []



patch_input = glob.glob('input_img/*.*')
for id, img in enumerate(patch_input):
    inp = skimage.io.imread(img)
resized_input = cv2.resize(inp, (64, 64))
res_general = cv2.resize(resized_input, (640, 640), interpolation=cv2.INTER_NEAREST)
inp_img = resized_input.reshape(resized_input.shape[0] * resized_input.shape[1], 3)
inp_img_list.append(inp_img)

ClusterCentersPIC1 = np.load('ClusterCentersPIC1.npy')


def sigma(ak, kmeans):
    list_count_pix = []
    for id4, img in enumerate(ak):
        s_cl = kmeans.predict(img)
        s_list = s_cl.tolist()
        s_list_count_pix = []
        i = list(range(clusters_n))
        for e in i:
            count_pix = s_list.count(e)
            s_list_count_pix.append(count_pix)
            if e == len(i):
                break
        list_count_pix.append(s_list_count_pix)
        if id4 == len(ak):
            break
    return list_count_pix, s_list

count_all_,s_list1 = sigma(all_pix, kmeans_s)
np.save('count_all_', count_all_)
count_all_= np.load('count_all_.npy')

count_input, s_list= sigma(inp_img_list, kmeans_s)


for  pix_col, cl_col in  zip(range(len(inp_img)), s_list):
    inp_img[pix_col] = ClusterCentersPIC1[cl_col]
    uj=inp_img.reshape(64,64,3)
    # inp_img_list.putpixel(pix_col, ClusterCentersPIC1[cl_col])
neww = cv2.resize(uj,(640,640),interpolation=cv2.INTER_NEAREST)
new_im = matplotlib.image.imsave('new_img.jpeg', neww)



print(count_input)


subtraction_list = []
for iu, count in enumerate(count_all_):
    result = []
    for i, j in zip(count_input[0], count_all_[iu]):
        subtraction = i - j
        result.append(abs(subtraction))
    # subtraction = count_input[0]-count_all_[iu]
    subtraction_list.append(sum(result))
    if iu == len(count_all_):
        break


modul= sorted(subtraction_list)
index_modul = []
for fr, op in enumerate(modul):
    index_m = subtraction_list.index(op)
    index_modul.append(index_m)
    if fr == len(modul):
        break


list_same_i= []
for m, i_m in enumerate(index_modul):  # resize, add text=image title
    img_same = img_list[i_m]
    list_same_i.append(img_same)
    if m == 10:
        break
lkf= np.concatenate(list_same_i, axis=1)
fig, ax = plt.subplots()
# plt.figure()
barlist = plt.bar(range(int(len(count_input[0]))), count_input[0])  # строим гистограмму ( колонки цвета кластера, высота = количеству пикселей в кластере)
for i, Color in enumerate(ClusterCentersPIC1):
    barlist[i].set_color(np.append(Color, 255)/255.)

plt.show()
plt.draw()
fig.savefig('plot_input.jpeg')


resized_image_s = cv2.resize(lkf, (2816, 256))
img_read_plot = matplotlib.image.imread('plot_input.jpeg')
resized_image_plot1 = cv2.resize(img_read_plot, (256, 256))

okkm = np.concatenate((resized_image_plot1, cv2.resize(inp, (256, 256)), resized_image_s), axis=1)
m_im = matplotlib.image.imsave('this_img.jpeg', okkm)
print(list_same_i)
img_read_plot1 = matplotlib.image.imread('plot_input.jpeg')
resized_image_plot = cv2.resize(img_read_plot1, (640, 640))

iku = np.concatenate((resized_image_plot, res_general, neww ), axis=1)
test_img = matplotlib.image.imsave('test_img.jpeg', iku)
print(res_general)