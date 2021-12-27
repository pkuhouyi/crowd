

import os
import numpy as np
from PIL import Image
from random import shuffle
import random
from tensorflow.keras.applications.imagenet_utils import preprocess_input


# 输入是ndarray数据格式，分别是原图，密度图和locations二维点图
# 他们的大小都是一样的，都是numpy的格式
# 生成10倍于原图的数据，原图+4乘corner+4个随机位置
def generate_slices(images, ada_densitys, locations):
    out_images = []
    out_ada_densitys = []
    out_locations = []
    for i, img in enumerate(images):
        out_images.append(img)
        out_ada_densitys.append(ada_densitys[i])
        out_locations.append(locations[i])
        try:
            img_h, img_w, _ = img.shape
        except BaseException:
            print('n-dim exception occuer!!!')
            continue
        weight = img_w // 2
        height = img_h // 2
        for w_index in range(2):
            for h_index in range(2):
                px = w_index * weight
                py = h_index * height
                out_images.append(img[py:py + height, px:px + weight])
                out_ada_densitys.append(ada_densitys[i][py:py + height, px:px + weight])
                out_locations.append(locations[i][py:py + height, px:px + weight])
        px = img_w // 2 - weight // 2
        py = img_h // 2 - height // 2
        out_images.append(img[py:py + height, px:px + weight])
        out_ada_densitys.append(ada_densitys[i][py:py + height, px:px + weight])
        out_locations.append(locations[i][py:py + height, px:px + weight])
        px_max = weight
        py_max = height
        for w_index in range(2):
            for h_index in range(2):
                px = random.randint(0, px_max - 1)
                py = random.randint(0, py_max - 1)
                out_images.append(img[py:py + height, px:px + weight])
                out_ada_densitys.append(ada_densitys[i][py:py + height, px:px + weight])
                out_locations.append(locations[i][py:py + height, px:px + weight])
    return (out_images, out_ada_densitys,out_locations)


# 输入是ndarray数据格式，分别是原图，密度图和locations二维点图
# 他们的大小都是一样的，都是numpy的格式
# 向左和右方向反转阵列
# 生成2倍于原图的数据量
def flip_slices(images, ada_densitys,locations):
    out_images = []
    out_ada_densitys = []
    out_locations = []
    for i, img in enumerate(images):
        # original
        out_images.append(img)
        out_ada_densitys.append(ada_densitys[i])
        out_locations.append(locations[i])
        # flip: left-right
        out_images.append(np.fliplr(img))  # 向左和右方向反转阵列
        out_ada_densitys.append(np.fliplr(ada_densitys[i]))
        out_locations.append(np.fliplr(locations[i]))
    return (out_images, out_ada_densitys,out_locations)

# 输入是ndarray数据格式，分别是原图，密度图和locations二维点图
# 他们的大小都是一样的，都是numpy的格式
# 洗牌操作
def shuffle_slices(images, ada_densitys, locations):
    out_images = []
    out_ada_densitys = []
    out_locations = []
    index_shuf = list(range(len(images)))  # python3中特加list强转
    shuffle(index_shuf)  # 这个函数用于洗牌
    for i in index_shuf:
        out_images.append(images[i])
        out_ada_densitys.append(ada_densitys[i])
        out_locations.append(locations[i])
    return (out_images, out_ada_densitys, out_locations)


# 输入是ndarray数据格式，分别是原图，密度图和locations二维点图
# 他们的大小都是一样的，都是numpy的格式
# 这里生成times倍的原图的数据量增强
# time表明做多少次pca
# 这里的随机数参数选取的是0.2
def pca_augmentation(images, ada_densitys, locations, times=1):  # time表明做多少次pca
    out_images = []
    out_ada_densitys = []
    out_locations = []
    for i, img in enumerate(images):
        out_images.append(img)
        out_ada_densitys.append(ada_densitys[i])
        out_locations.append(locations[i])
        for time in range(times):
            rand = []
            rand.append(random.gauss(0, 0.2))
            rand.append(random.gauss(0, 0.2))
            rand.append(random.gauss(0, 0.2))
            _img = img.reshape(img.shape[2], img.shape[0], img.shape[1]) / 255.0

            image = np.zeros(_img.shape, dtype=np.float)

            value = np.array([0.00579212, 0.01123351, 0.21004208])

            vector = np.array([(0.1182795, -0.79894806, -0.5896541), (-0.76679834, 0.30379899, -0.56544361),
                               (0.63089639, 0.51902617, -0.57669886)])
            added = np.matmul(vector, (value * np.array(rand)).T)
            image[0] = _img[0] + added[0]
            image[1] = _img[1] + added[1]
            image[2] = _img[2] + added[2]
            image = image.reshape(img.shape)
            ma = image.max()
            image = image / ma
            image = (image.reshape(img.shape) * 255).astype(np.uint8)

            out_images.append(image)
            out_ada_densitys.append(ada_densitys[i])
            out_locations.append(locations[i])

    return out_images, out_ada_densitys, out_locations



# 输入时numpy格式的2维密度图
# resize 密度图到1/para，在这里乘了系数，使得总人数没有发生变化
# 返回resize之后的密度图
def resize_density(density,para=8):
    sum1=np.sum(density)
    den = Image.fromarray(density)
    x, y = den.size
    temp = den.resize((x // para, y // para), Image.BILINEAR)
    post_density=np.array(temp)
    sum2=np.sum(post_density)
    if sum2==0:
        return post_density
    return post_density * sum1/sum2


# resize dotmap到指定的倍数
# 值得注意的是由于将采样的时候可能产生点与点的重叠，
# 这里采用直接累加点值的方法
def resize_dotmap(dot_map,para=8):
    pts = np.array(list(zip(np.nonzero(dot_map)[0], np.nonzero(dot_map)[1])))
    shape=(dot_map.shape[0] // para,dot_map.shape[1] // para)
    post_dotmap = np.zeros(shape, dtype=np.float32)
    for i, pt in enumerate(pts):
        x=int(round(pt[0]/para))
        y=int(round(pt[1]/para))
        try:
            post_dotmap[x,y] += 1.  # 一副图片只有这个点为255
        except Exception:
            post_dotmap[x-1,y-1] = 1.
    return post_dotmap


def save_npy(npy_base_path,images, ada_densitys, locations):
    if not os.path.exists(npy_base_path):
        os.makedirs(npy_base_path)
    for i,img in enumerate(images):
        npy_name=npy_base_path + str(i) + '.npy'
        img = preprocess_input(img, mode='tf')
        ada=resize_density(ada_densitys[i])
        loc=resize_dotmap(locations[i])
        np.save(npy_name, (img,ada,loc))






