import colorsys
import glob
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib
import cv2
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



def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))



def clear_name(docname,
               slash_replace='-',  # слэш: заменять на минус; используется в идентификаторах документов: типа № 1/2
               quote_replace='',  # кавычки: замены нет - удаляем
               multispaces_replace='\x20',  # множественные пробелы на один пробел
               quotes="""“”«»'\""""  # какие кавычки будут удаляться
               ):
    docname = re.sub(r'[' + quotes + ']', quote_replace, docname)
    docname = re.sub(r'[/]', slash_replace, docname)
    docname = re.sub(r'[|*?<>:\\\n\r\t\v.jpegims]', '', docname)  # запрещенные символы в windows
    docname = re.sub(r'\s{2,}', multispaces_replace, docname)
    docname = docname.strip()
    docname = docname.rstrip('-')  # на всякий случай
    docname = docname.rstrip('.')  # точка в конце не разрешена в windows
    docname = docname.strip()  # не разрешен пробел в конце в windows
    docname = re.split("_", docname)

    return docname[1]

# with open('AppIconsCSV.csv') as file:
#     reader = csv.reader(file)     #читаем файл со всеми адресами картинок
#     my_list_url = list(map(tuple, reader))
# for id, image_filename in enumerate(my_list_url):
#     image_numpy = skimage.io.imread(image_filename[0])
#     im = Image.fromarray(image_numpy)
#     im.save("imgs" + os.sep + str(id).zfill(5) + ".jpeg")
#     if id==1000:
#         break
#
# print('DONE!')

# my_listS = []


clusters_n = 5

# def cluster_img(path): #функция для получения кластеров картинок
#     a_list = []
#     pic_list = []
#     for id, img in enumerate(glob.glob(path)):
#         a = matplotlib.image.imread(img)
#         A = a.reshape(a.shape[0]*a.shape[1], 3)
#         a_list.append(A)
#     # pic=a.tolist()
# # my_listS.append(a)
# # my_listSave = np.array(my_listS)
# # X = my_listSave.reshape(my_listSave.shape[0]*my_listSave.shape[1]*my_listSave.shape[2], 3)
# # kmeans_s = KMeans(n_clusters=10).fit(X)
# # ClusterCenters = np.array(kmeans_s.cluster_centers_, dtype="uint8")
# #     for pixel2 in pic:
# #         pixel=np.array(pixel2)
#         # pixel=pixel1.reshape(pixel1.shape[0]*pixel1.shape[1]*pixel1.shape[2], 3)
#         global kmeans_s
#         kmeans_s = KMeans(n_clusters=int(clusters_n)).fit(A)
#         ClusterCentersPIC = np.array(kmeans_s.cluster_centers_, dtype="uint8")
#
#         fig, ax = plt.subplots()
#         barlist = plt.bar(range(0,5),range(0,5))  # строим гистограмму ( колонки цвета кластера, высота = количеству пикселей в кластере)
#         for i, Color1 in enumerate(ClusterCentersPIC):
#             barlist[i].set_color(np.append(Color1, 255)/255.)
#         plt.show()
#
#
#         hsv = []
#         for cl_item in ClusterCentersPIC:
#             el_hsv = []
#             r, g, b = (cl_item/255)
#             h, s, v = colorsys.rgb_to_hsv(r, g, b)
#             el_hsv.append(h)
#             el_hsv.append(s)
#             el_hsv.append(v)
#             hsv.append(el_hsv)
#         pic_list.append(hsv)
#         if id == len(glob.glob(path)):
#             break
#     np_pic_list = np.array(pic_list)
#
#     fig, ax = plt.subplots()
#     barlist1 = plt.bar(range(0, 5), range(0,
#                                          5))  # строим гистограмму ( колонки цвета кластера, высота = количеству пикселей в кластере)
#     for ie, Color2 in enumerate(np_pic_list[0]):
#         barlist1[ie].set_color(Color2)
#     plt.show()
#
#     a_nparr = np.array(a_list)
#     return np_pic_list, kmeans_s, a_nparr



def cluster_img(path): #функция для получения кластеров картинок
    a_list = []
    pic_list = []
    for id, img in enumerate(glob.glob(path)):
        a = matplotlib.image.imread(img)
        A = a.reshape(a.shape[0]*a.shape[1], 3)
        a_list.append(A)
        global kmeans_s
        kmeans_s = KMeans(n_clusters=int(clusters_n)).fit(A)
        ClusterCentersPIC = np.array(kmeans_s.cluster_centers_, dtype="uint8")
        pic_list.append(ClusterCentersPIC)
        # fig, ax = plt.subplots()
        # barlist = plt.bar(range(0,5),range(0,5))  # строим гистограмму ( колонки цвета кластера, высота = количеству пикселей в кластере)
        # for i, Color1 in enumerate(ClusterCentersPIC):
        #     barlist[i].set_color(np.append(Color1, 255)/255.)
        # plt.show()


        # hsv = []
        # for cl_item in ClusterCentersPIC:
        #     el_hsv = []
        #     r, g, b = (cl_item/255)
        #     h, s, v = colorsys.rgb_to_hsv(r, g, b)
        #     el_hsv.append(h)
        #     el_hsv.append(s)
        #     el_hsv.append(v)
        #     hsv.append(el_hsv)
        # pic_list.append(hsv)
        # if id == len(glob.glob(path)):
        #     break
    np_pic_list = np.array(pic_list)

    # fig, ax = plt.subplots()
    # barlist1 = plt.bar(range(0, 5), range(0,
    #                                      5))  # строим гистограмму ( колонки цвета кластера, высота = количеству пикселей в кластере)
    # for ie, Color2 in enumerate(np_pic_list[0]):
    #     barlist1[ie].set_color(Color2)
    # plt.show()

    a_nparr = np.array(a_list)
    return np_pic_list, kmeans_s, a_nparr


# def min_d(matrix):  # function to find minimum values
#     af = []  # in the columns of the matrix of distances of clusters of the main pictures with many pictures
#     for x5 in range(len(matrix)):
#         list_val_col = [matrix[ir][x5] for ir in range(0, int(clusters_n))]
#         z = min(list_val_col)
#         af.append(z)
#         if x5 == len(matrix):
#             break
#     mean_d = np.mean(af)
#     return mean_d


def sigma(ak, kmeans):
    list_count_pix = []
    # if ak.shape == (1, 262144, 3):
    #     ak == ak
    # else:
    #     ak == ak[2]
    for id4, img in enumerate(ak):
        # dr = TruncatedSVD()  # count the number of pixels in each SIGMA cluster
        # dr.fit(img)
        # s_dr = dr.transform(img)
        # kmeans.fit(s_dr)
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
    return list_count_pix


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"



main_picture, km_mp, a_mp = cluster_img('main_img/*.*')  # find clusters of base images
Sigma_main_picture = sigma(a_mp, km_mp)
# number of pixels in the main image in cluster
smp_sorted = sorted(Sigma_main_picture[0], reverse=True)
Sigma_main_pic=Sigma_main_picture[0]
sum_smp = []
for n1 in range(len(smp_sorted)):
    sum_smp.append(smp_sorted[n1])
    if sum(sum_smp) >= sum(Sigma_main_pic)/2:
        break
index_max_cl = []
for e1, m1 in enumerate(sum_smp):
    for t1, rt in enumerate(Sigma_main_pic):
        if rt == m1:
            in_smp = t1
            index_max_cl.append(int(in_smp))
            if t1 == len(Sigma_main_pic):
                break
            if e1 == len(sum_smp):
                break
main_pictur = main_picture[0]
max_main_picture=[]
for ti, yu in enumerate(index_max_cl):
    mpe = main_pictur[yu]
    max_main_picture.append(mpe)
    if ti == len(index_max_cl):
        break




# index_max_all_pic = []
# Sigma_list = []  # matrix 10x10 of quantity C max / C min
# for idd, smp in enumerate(Sigma_main_picture):
#     s1=max(smp)
#     global i1
#     i1= np.where(s1)
#     for y, sap in enumerate(Sigma_all_pic):
#         s = np.zeros((int(clusters_n), int(clusters_n)), dtype="float32")
#         list_ss_matrix = []
#         # for id9, s1 in enumerate(smp):  # consider Smax/Smin(s-number of pixels in each cluster)=sigma and get the matrix
#         list_ss = []
#             # for ii, s2 in enumerate(max(sap)):
#                 # s_total = [s1, s2]
#         # s_total = [smp, max(sap)]
#         s2 = max(sap)
#         ii2= np.where(s2)
#         index_max_all_pic.append(ii2)
#         s_total = [s1, s2]
#         s = max(s_total) / min(s_total)
#
#         list_ss.append(s)
#             # if id9 == len(sap):
#             #     break
#         list_ss_matrix.append(list_ss)
#         if y == len(Sigma_all_pic):
#             break
#         Sigma_list.append(list_ss_matrix)
#     if idd == 1:
#     # if idd == len(max(Sigma_main_picture)):
#         break
all_pic, km_ap, a_ap = cluster_img('imgs/*.*')  # find clusters of a set of pictures
Sigma_all_pic = sigma(a_ap, km_ap)  # number of pixels in all pictures in cluster

np.save('Sigma_all_pic', Sigma_all_pic)
Sigma_all_pic = np.load('Sigma_all_pic.npy')

index_max_all_pic = []
Sigma_list = []  # matrix 10x10 of quantity C max / C min
for y, sap in enumerate(Sigma_all_pic):
    sig_list = []
    for id9, s1 in enumerate(sum_smp):
        s = np.zeros((int(len(sum_smp)), int(clusters_n)), dtype="float32")
        list_ss = []
        for ii, s2 in enumerate(sap):
            s_total = [s1, s2]
            s = max(s_total) / min(s_total)
            list_ss.append(s)
            if ii == len(sap):
                break
        sig_list.append(list_ss)  # matrix 2 imdgs 5x5
        if id9 == len(sum_smp):
            break
    Sigma_list.append(sig_list)
    if y == len(Sigma_all_pic):
        break


max_main_picture_np = np.array(max_main_picture)
dd = []
for ii, pici in enumerate(all_pic):  # we find the study m \ y the main picture and the rest of the population DELTA
    # d1 = []
    d = np.zeros((int(len(max_main_picture_np)), int(clusters_n)), dtype='float32')
    for id7, m_p in enumerate(max_main_picture_np):
        for j in range(0, int(clusters_n)):
            d[id7][j] = euclidean_distance(m_p[id7], pici[j])
    # d = euclidean_distance(m_p, pici)
    dd.append(d)
    if id7 == len(max_main_picture_np):
        break
        if ii == len(all_pic):
            break


# dd = []
# for id7, m_p in enumerate(main_picture):
#     m_p=m_p[i1]
#     for ii, pici in enumerate(all_pic):  # we find the study m \ y the main picture and the rest of the population DELTA
#         pici = pici[ii2]
#         d = np.zeros((int(clusters_n), int(clusters_n)), dtype='float32')
#         d = euclidean_distance(m_p, pici)
#         dd.append(d)


# min_list=[]
# for id , i in enumerate(dd):
#     s = min_d(i)
#     min_list.append(s)
#     if id==len(dd):
#         break


# fig, ax = plt.subplots()
# barlist = plt.bar(range(len(Sigma_main_picture[0])), sorted(Sigma_main_picture[0]))   # строим гистограмму ( колонки цвета кластера, высота = количеству пикселей в кластере)
#
# for i, Color in enumerate(main_picture[0]):
#     barlist[i].set_color(Color)
# plt.show()
# plt.savefig('main.png')

# np.save('sigma_list', Sigma_list)
# Sigma_list = np.load('sigma_list.npy')

np_sigma = np.array(Sigma_list)  #.reshape(int(len(Sigma_all_pic)), int(clusters_n), int(clusters_n))
print('sigma=', np_sigma)
np_dd = np.array(dd)
DxS_list = []
for y, i_dd in enumerate(np_dd):
    for id1, j_s in enumerate(np_sigma):
        # DxS = j_s.dot(i_dd)# multiply the matrix with distances by the matrix with divided quantities
        # i_dd.reshape(5, 2)
        # DxS = np.dot(j_s, i_dd.reshape(5, 2))
        DxS = np.zeros((int(i_dd.shape[0]), int(i_dd.shape[1])), dtype="float32")
        for b1, bn in enumerate(range(0, i_dd.shape[0])):
            for b2, bf in enumerate(range(0, i_dd.shape[1])):
                DxS[b1][b2] = j_s[b1, b2] * i_dd[b1, b2]
                if b2 == i_dd.shape[1]:
                    break
            if b1 == i_dd.shape[0]:
                break
        DxS_list.append(DxS)
        if id1 == id1:
            break
    if y == len(np_dd):
        break
np.save('ds_list', DxS_list)
DxS_list = np.load('ds_list.npy')


list_min_sigma = []
list_min_sigma_index = []
for id3, el in enumerate(DxS_list):  # find pairs in the matrix by the minimum values of the product delta x sigma
    min_sigma = []
    index_s = []
    #i1 == 0
    for i1 in range(0, int(len(max_main_picture_np))):
        y = np.amin(el)
        index_y, index_x = np.where(el == y)
        min_sigma.append(y)
        y += 1000
        el[index_y, :] += 1000
        el[:, index_x] += 1000
        el[index_y, index_x] = y
        index_s.append(index_x)
        if i1 == len(range(0, int(len(max_main_picture_np)))):
            break
    list_min_sigma.append(min_sigma)
    list_min_sigma_index.append(index_s)
    if id3 == len(DxS_list):
        break

# np.save('list_min_sigma', list_min_sigma)
# list_min_sigma = np.load('list_min_sigma.npy')
# np.save('list_min_sigma_index', list_min_sigma_index)
# list_min_sigma_index = np.load('list_min_sigma_index.npy')
np_sum_smp = np.array(sum_smp)
np_Sigma_main_pic = np.array(Sigma_main_pic)
list_total = []   # we calculate the sum of c1 and c2 10 pairs (which have the minimum sigma by index)
for s2, ind in zip(Sigma_all_pic, list_min_sigma_index):
    np_ind = np.array(ind)
    total_s = []
    # for f, index in enumerate(np_ind):
    #     for tyt, u in enumerate(range(0, len(np_sum_smp))):
    for index, u in zip(np_ind, range(0, len(np_sum_smp))):
        total_el = np_sum_smp[u]+s2[index[0]]
        total_s.append(int(total_el))
        #     if u == len(range(0, len(np_sum_smp))):
        #         break
        # if f == len(np_ind):
        #     break
    list_total.append(total_s)

list_total_summa = []
for er, to in enumerate(list_total):
    total_summa= sum(to)
    list_total_summa.append(total_summa)
    if er == len(list_total):
        break

# np.save('list_total', list_total)
# list_total = np.load('list_total.npy')

list_result = []
for id6, summa in enumerate(list_total_summa):
    result = summa/524288
    list_result.append(result)
    if id6 == len(list_total_summa):
        break

list_1_x = []
for sig, sf in zip(list_min_sigma, list_result):  # multiply "softmax" by the product of delta by sigma
    # np_sig == np.array(sig)
    for ah, para in enumerate(sig):
        list_dxsxsm = []
        # for x, y in zip(range(0, int(len(max_main_picture_np))), range(0, int(len(max_main_picture_np)))):
        dxsxsm = para * sf
        list_dxsxsm.append(dxsxsm)
        if ah == len(sig):
            break
    list_1_x.append(list_dxsxsm)

list_total_sum = []
for id11, sum_s in enumerate(list_1_x):
    total_sum = np.sum(sum_s)
    list_total_sum.append(total_sum)
    if id11 == len(list_1_x):
        break

# dir_pic = 'imgs/*.*'
# old_list = []
# for i, y in enumerate(glob.glob(dir_pic)):   # read current image names to rename them to new ones
#     old_list.append(y)
#     if i == len(glob.glob(dir_pic)):
#         break
# for id7, i in enumerate(range(len(list_total_sum))):        # renаme imgs name= list_total_sum
#     # os.mkdir("imgs_new1")
#     # shutil.copyfile("D:\\test.txt", "D:\\test2.txt")
#     new_file = os.path.join("imgs" + os.sep+str(toFixed(list_total_sum[i], 5)).zfill(10) + str("_") + str(clear_name(old_list[id7])) + ".jpeg")
#     old_file = os.path.join(old_list[i])
#     os.rename(old_file, new_file)
#     if id7 == len(list_total_sum):
#         break

min_max = sorted(list_total_sum)
list_total_index = []
for id18, m_m in enumerate(min_max):
    total_index = list_total_sum.index(m_m)
    # if total_index in list_total_index:
    #     total_index_next = list_total_sum[total_index:].index(m_m)
    list_total_index.append(total_index)
    if id18 == len(min_max):
        break



same_img = list_total_index[0:19]  # list of most similar pictures
n = 20
diff_img = sorted(list_total_index[:-n-1:-1])  # list of the most dissimilar pictures
mean_ind = len(list_total_index)//2
mean_img= (list_total_index[(int(mean_ind)-10):(int(mean_ind)+10)])

resize_list_same = []
path_imgs = 'imgs/*.*'
for mj, nh in enumerate(same_img):
# for i, img_el in enumerate(glob.glob(path_imgs)[0:20]):  # resize, add text=image title
    img_el = glob.glob(path_imgs)[nh]
    img_read = matplotlib.image.imread(img_el)
    resized_image = cv2.resize(img_read, (256, 256))
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(resized_image, str(img_el), (30, 15), font, 1, color=(0, 255, 0), thickness=2)
    resize_list_same.append(resized_image)
    if mj == 20:
        break


resize_list_same = []
for id19, num in enumerate(same_img):
    resized_image = cv2.resize(a_ap[num].reshape(512,512,3), (256, 256))
    resize_list_same.append(resized_image)
    if id19 == len(same_img):
        break

resized_image4 = []
resized_image1 = cv2.resize(a_mp.reshape(512, 512, 3), (236, 236))
bordersize = 10
resized_image3 = cv2.copyMakeBorder(resized_image1, top=bordersize,
                                   bottom=bordersize, left=bordersize,
                                   right=bordersize, borderType=cv2.BORDER_CONSTANT,
                                   value=[255, 0, 0])
resized_image4.append(resized_image3)
x1234 = np.array(resized_image4 + resize_list_same)
viss = np.concatenate(x1234, axis=1)
s_i = matplotlib.image.imsave('same_img.jpeg', viss)

resize_list_diff = []
path_imgs = 'imgs/*.*'
for tg, yh in enumerate(diff_img):
# for i, img_el in enumerate(glob.glob(path_imgs)[:-n-1:-1]):  # resize, add text=image title
    img_el = glob.glob(path_imgs)[yh]

    img_read = matplotlib.image.imread(img_el)
    resized_image2 = cv2.resize(img_read, (256, 256))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resized_image, str(img_el), (30, 15), font, 1, color=(0, 0, 255), thickness=2)
    resize_list_diff.append(resized_image2)
    if tg == 20:
        break

resize_list_diff = []
for id20, num1 in enumerate(diff_img):
    resized_image5 = cv2.resize(a_ap[num1].reshape(512,512,3), (256, 256))
    resize_list_diff.append(resized_image5)
    if id20 == len(diff_img):
        break
diff_12345 = np.array(resize_list_diff)
viz = np.concatenate(diff_12345, axis=1)
d_i = matplotlib.image.imsave('diff_img.jpeg', viz)

# resize_list_mean = []
# for id21, num2 in enumerate(mean_img):
#     resized_image6 = cv2.resize(a_ap[num2].reshape(512,512,3), (256, 256))
#     resize_list_mean.append(resized_image6)
#     if id21 == len(mean_img):
#         break
# mean_12345 = np.array(resize_list_mean)
# vim = np.concatenate(mean_12345, axis=1)
# m_i = matplotlib.image.imsave('mean_img.jpeg', vim)

img_read_diff = matplotlib.image.imread('diff_img.jpeg')
resized_image_diff = cv2.resize(img_read_diff, (5120, 256))
img_read_same = matplotlib.image.imread('same_img.jpeg')
resized_image_same = cv2.resize(img_read_same, (5120, 256))
# img_read_mean= matplotlib.image.imread('mean_img.jpeg')
# resized_image_mean= cv2.resize(img_read_mean, (5120, 256))

viz_g = np.concatenate((resized_image_same, resized_image_diff), axis=0)   # glue pictures in 1
g_i = matplotlib.image.imsave('general_img.jpeg', viz_g)
# papka = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'imgs_new1')
# shutil.rmtree(papka)

# for row in my_list_distance:   # draw a histogram
#     for i in range(0,10):
#         print(*row)
# from random import randint
# N = 10
# M = 10
# lst=[[randint(10, 10) for i in range(N)] for i in range(M)]
# for i in lst:
#     print()
#     for j in i:
#         print (j, end=" ")

# fitting
# dr = TruncatedSVD()      #считаем количество пикселей в каждом кластре
# dr.fit(X)
# x_dr = dr.transform(X)

# kmeans.fit(x_dr)
# y = kmeans.predict(x_dr)
# y_list=y.tolist()
# list_count_pix=[]
# i= list(range(0, 10, 1))
# for e in i:
#     count_pix=y_list.count(e)
#     list_count_pix.append(count_pix)
#     if e==9:
#         break


# array = np.expand_dims(ClusterCenters, 0)
# im=matplotlib.image.imsave('image1.jpeg', array)      # получаем картинку из  пикселей всех кластеров
# q=array.reshape(10,3)


# for id, w in enumerate(q):
#     #W=w.reshape(-1,1)
#     filename = 'element\element'+'_'+ str(id)+'.jpeg'
#     # if os.path.isfile(filename):
#     #     os.remove(filename)
#     f=matplotlib.image.imsave(filename, np.array([[w]]).astype('uint8'))
#     if id==9:
#         break


# my_list2 = []
# path2 = 'element/*.*'
# for img2 in glob.glob(path2):
#     g = matplotlib.image.imread(img2)
#     resized_image = cv2.resize(g, (100, 100))      # изменяем размер пикселей кластеров, увеличиваем в 100 раз
#     my_list2.append(resized_image)
#
# my_list3=np.array(my_list2)
# for id,q in enumerate(my_list3):
#         filename = 'el100\el100' + '_' + str(id) + '.jpeg'
#         if os.path.isfile(filename):
#             os.remove(filename)
#         g = matplotlib.image.imread(img2)
#         RR = matplotlib.image.imsave(filename, q)
#         if id==9:
#             break

# array100=np.array(my_list3)
# vis = np.concatenate(array100, axis=1)   #получаем картинку из увеличенных  пикселей 100х1000
# pp=matplotlib.image.imsave('out.jpeg', vis)
#
#
# fig, ax = plt.subplots()
# barlist = plt.bar(range(len(list_count_pix)), sorted(list_count_pix))   # строим гистограмму ( колонки цвета кластера, высота = количеству пикселей в кластере)
#
# for i, Color in enumerate(ClusterCenters):
#     barlist[i].set_color(np.append(Color, 255)/255.)
# plt.show()
# plt.savefig('demo.png')