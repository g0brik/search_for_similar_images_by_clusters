import csv
import glob
import os
import numpy as np
import skimage
import cv2
import pandas
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from numpy.random import rand
import sklearn
import pandas as pd
from random import randint
from itertools import groupby


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

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


def Cluster_img(path): #функция для получения кластеров картинок
    pic_list=[]
    for id,img in enumerate(glob.glob(path)):
        a = matplotlib.image.imread(img)
        A=a.reshape(a.shape[0]*a.shape[1], 3)
    # pic=a.tolist()
# my_listS.append(a)
# my_listSave = np.array(my_listS)
# X = my_listSave.reshape(my_listSave.shape[0]*my_listSave.shape[1]*my_listSave.shape[2], 3)
# kmeans = KMeans(n_clusters=10).fit(X)
# ClusterCenters = np.array(kmeans.cluster_centers_, dtype="uint8")
#     for pixel2 in pic:
#         pixel=np.array(pixel2)
        # pixel=pixel1.reshape(pixel1.shape[0]*pixel1.shape[1]*pixel1.shape[2], 3)
        kmeans = KMeans(n_clusters=10).fit(A)
        ClusterCentersPIC = np.array(kmeans.cluster_centers_, dtype="uint8")
        pic_list.append(ClusterCentersPIC)
        if id==9:
            break
    return pic_list


def min_d(matrix):  # функция для нахождения минимальных значений в столбцах матрицы расстояний кластеров осн картинки с множеством картинок
    f=[]
    for x in range(0,10):
        list_val_col = [matrix[i][x] for i in range(0,10)]
        z=min(list_val_col)
        f.append(z)
        if x==9:
            break
    mean_d = np.mean(f)
    return mean_d


main_picture=Cluster_img('main_img/*.*')   #находим кластеры осн картинки
all_pic=Cluster_img('imgs/*.*') #находим кластеры множества картинок
print(main_picture)
print(all_pic)

dd=[]
for ii, pici in enumerate(all_pic):  #находим расстояния м\у основной картинкой и множеством ост
    d=np.zeros((10, 10), dtype="float32")
    for i in range(0,10):
        for j in range(0,10):
            # d[i][j] = np.linalg.norm(main_picture -pici[j]) #расчет рассояний
            d[i][j]=euclidean_distance(main_picture,pici[j])
    dd.append(d)


min_list=[]
for id, i in enumerate(dd):
    s=min_d(i)
    min_list.append(s)
    if id==len(dd):
        break
dir = 'imgs/*.*'
for id,i in enumerate(min_list):
    for id,img in enumerate(glob.glob(dir)):
        old_file = os.path.join(img)
        new_file = os.path.join("imgs" + os.sep + str(i).zfill(5) + "_img" + ".jpeg")
    os.rename(old_file, new_file)
    if i==len(min_list):
        break



min_max=sorted(min_list)
same_img=min_max[0:5]  #список самых похожих картинок
n=5
diff_img=min_max[:-n-1:-1]  #список самых непохожих картинок
print(same_img)
print(diff_img)




# for row in my_list_distance:
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
# dr = TruncatedSVD()
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
# im=matplotlib.image.imsave('image1.jpeg', array)
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
#     resized_image = cv2.resize(g, (100, 100))
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
# vis = np.concatenate(array100, axis=1)
# pp=matplotlib.image.imsave('out.jpeg', vis)
#
#
# fig, ax = plt.subplots()
# barlist = plt.bar(range(len(list_count_pix)), sorted(list_count_pix))
#
# for i, Color in enumerate(ClusterCenters):
#     barlist[i].set_color(np.append(Color, 255)/255.)
# plt.show()
# plt.savefig('demo.png')