import os
import glob
import math
import scipy
import scipy.io as sio
import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage
from abc import ABCMeta,abstractmethod
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.spatial

class BasePrepare(metaclass=ABCMeta):
    def __init__(self,k=4,max_sigma=100,para=0.3):
        self.k=k
        self.max_sigma=max_sigma
        self.para=para

    @abstractmethod
    def load_gt(self, gtfile_path):
        '''
        :param gtfile_path:gt文件路径
        :return: ndarray:(point_num,2),是每个人头点的位置x,y坐标,x是距离图像左边的距离，y是图像距离上边的距离
        '''
        pass

    def generate_dot_map(self, locations, dot_map_shape):
        '''
        生成一个二维的0，1人头点图,有人头的地方为255，其他的地方为0
        :param locations:ndarray(num_point,2)是self.load_gt返回的结果
        :param dot_map_shape:tuple:2 比较典型的是（768，1024）
        :return:ndarray(768,1024)，返回的顺序和PIL格式不同，这一点要注意，注意dtype是uint8，返回的值是1
        '''
        dot_map = np.zeros((dot_map_shape[1], dot_map_shape[0]), dtype='uint8')
        for dot in locations:
            try:
                x = int(math.floor(dot[1]))
                y = int(math.floor(dot[0]))
                dot_map[x, y] = 1
            except IndexError:
                print((dot_map_shape[1], dot_map_shape[0]))
        return dot_map

    def adaptive_gaussian_generator(self,dot_map):
        '''
        :param dot_map:ndarray(768,1024),是generate_dot_map返回的结果
        :param k:是k-1个最近的邻居节点的距离
        :param para:是sigma需要乘上的系数,控制高斯核标准差的系数
        :return:返回经过高斯核卷积之后的密度图，ndarray(768,1024)
        '''
        density_map = np.zeros(dot_map.shape, dtype=np.float32)  # 定义同样大小的密度图
        gt_count = np.count_nonzero(dot_map)  # 人头总数
        if gt_count == 0:
            return density_map
        pts = np.array(list(zip(np.nonzero(dot_map)[0], np.nonzero(dot_map)[1])))
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)  # 建立kdtree
        distances, locations = tree.query(pts, k=self.k)  # 寻找每个节点与自己最近的K个邻
        for i, pt in enumerate(pts):
            pt2d = np.zeros(dot_map.shape, dtype=np.float32)
            pt2d[pt[0], pt[1]] = 1.  # 一副图片只有这个点为1
            if gt_count > 1:
                sigma = np.average(distances[i][1:])  # 与第i个节点距离最近的节点
                if sigma > self.max_sigma:
                    sigma = self.max_sigma
            else:
                sigma = np.average(np.array(dot_map.shape)) / 2. / 2.  # case: 1 point
            sigma = sigma * self.para
            density_map += ndimage.filters.gaussian_filter(pt2d, sigma)
        return density_map

    @staticmethod
    def draw_head_dot(image_file, gt_file, radius=5, show=True):
        '''
        :param image_file:图像的路径
        :param gt_file:对应的gt的路径
        :param radius:默认即可，是圆圈的半径
        :param show:是否默认显示
        :return:返回融合之后的PIL格式的图像
        '''
        img = Image.open(image_file, mode='r').convert("RGB")
        locations = BasePrepare.load_gt(gt_file)
        gt = Image.new("RGB", img.size, "blue")
        drawObject = ImageDraw.Draw(gt)
        for i in locations:
            drawObject.ellipse((i[0], i[1], i[0] + radius, i[1] + radius), fill="red")  # 在image上画一个红色的圆
        result_img = Image.blend(img, gt, 0.3)
        if show:
            result_img.show()
        return result_img


    @staticmethod
    def draw_density_dot_heatmap(image,density_map,dot_map):
        '''
        :param image:ndarray 2维的numpy格式的原图
        :param density_map:ndarray
        :param dot_map:ndarray
        :return:
        '''
        img = Image.fromarray(image.astype('uint8')).convert('RGB')
        plt.figure('show')
        plt.subplot(131)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(132)
        cmap = cm.get_cmap('jet',1000)
        plt.imshow(density_map, cmap=cmap, extent=None, aspect='equal')
        plt.axis('off')
        sum=np.sum(density_map)
        print('density_count:',sum)
        plt.subplot(133)
        plt.imshow(dot_map,extent=None, aspect='equal')
        plt.axis('off')
        sum = np.sum(dot_map)/255
        print('true_count:', sum)
        plt.show()

    def adaptive_radius(self, locations):
        '''
        :param locations:
        :return:
        '''
        leafsize = 2048
        tree = scipy.spatial.KDTree(locations.copy(), leafsize=leafsize)  # 建立kdtree
        distances, locations = tree.query(locations, k=4, eps=10.)  # 寻找每个节点与自己最近的两个节点
        radius=[]
        for i, pt in enumerate(locations):
            sigma = np.average(distances[i][1:])  # 与第i个节点距离最近的节点
            if sigma > self.max_sigma:
                sigma = self.max_sigma
            sigma = sigma * self.para
            radius.append(sigma)
        return radius

    def draw_head_adaptive_theta_circle(self, image_file, gt_file):
        '''
        :param image_file:
        :param gt_file:
        :return:
        '''
        image = Image.open(image_file, mode='r').convert("RGB")
        locations = BasePrepare.load_gt(gt_file)
        radius=self.adaptive_radius(locations)  # 生成人头点图
        draw = Image.new("RGB", image.size, "blue")
        drawObject = ImageDraw.Draw(draw)
        for i,loc in enumerate(locations):
            drawObject.ellipse((loc[0]-radius[i]/2, loc[1]-radius[i]/2, loc[0] + radius[i]/2, loc[1] + radius[i]/2), fill="red")  # 在image上画一个红色的圆
        result = Image.blend(image, draw, 0.3)
        result.show()
        return result








