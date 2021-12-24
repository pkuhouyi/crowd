import os
import glob
import scipy.io as sio
import numpy as np
from PIL import Image
from datasets.basePrepare import BasePrepare


class ShhaiPrepare(BasePrepare):
    def __init__(self):
        super(ShhaiPrepare, self).__init__()


    def load_gt(self, gtfile_path):
        dataset = sio.loadmat(gtfile_path)
        dataset = dataset['image_info'][0, 0]
        locations = dataset['location'][0, 0]
        return locations


    def get_infos(self,image_file_name, gt_file_base_path):
        i = image_file_name.index('IMG_') + 4
        j = image_file_name.index('.jpg')
        substr = image_file_name[i:j]
        gt_file_path = gt_file_base_path + 'GT_IMG_' + substr + '.mat'
        return gt_file_path,substr

    def process_adaptive(self, path, ext='*.jpg',density_npy_name='/ada_gaussian_npy/',dot_npy_name='/dot_npy/'):
        images_path = path + '/images/'
        gts_path = path + '/ground-truth/'
        density_npy_path=path + density_npy_name
        dot_npy_path=path + dot_npy_name
        if not os.path.exists(density_npy_path):
            os.makedirs(density_npy_path)
        if not os.path.exists(dot_npy_path):
            os.makedirs(dot_npy_path)
        for i,image_file in enumerate(sorted(glob.glob(os.path.join(images_path, ext)))):
            print(image_file)
            image = Image.open(image_file, mode='r').convert("RGB")
            gt_file_path,substr=self.get_infos(image_file,gts_path)
            locations = self.load_gt(gt_file_path)
            dot_map = self.generate_dot_map(locations, image.size)
            desity_map = self.adaptive_gaussian_generator(dot_map)

            np.save(os.path.join(density_npy_path,substr + '.npy'),desity_map)
            np.save(os.path.join(dot_npy_path,substr + '.npy'),dot_map)




if __name__=="__main__":
    shhai=ShhaiPrepare()
    # a.draw_head_dot('/home/houyi/datasets/ShanghaiTech/part_B/train_data/images/IMG_1.jpg','/home/houyi/datasets/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_1.mat')
    shhai.process_adaptive('/home/houyi/datasets/ShanghaiTech/part_B/train_data')











