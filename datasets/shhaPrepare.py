import os
import glob
import scipy.io as sio
import numpy as np
from PIL import Image
from datasets.basePrepare import BasePrepare
import multiprocessing

class ShhaiPrepare(BasePrepare):
    def __init__(self,path, ext='*.jpg',density_npy_name='/ada_gaussian_npy/',dot_npy_name='/dot_npy/'):
        super(ShhaiPrepare, self).__init__()
        images_path = path + '/images/'
        self.gts_path = path + '/ground-truth/'
        self.density_npy_path=path + density_npy_name
        self.dot_npy_path=path + dot_npy_name
        if not os.path.exists(self.density_npy_path):
            os.makedirs(self.density_npy_path)
        if not os.path.exists(self.dot_npy_path):
            os.makedirs(self.dot_npy_path)

        self.image_files=sorted(glob.glob(os.path.join(images_path, ext)))

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

    def non_multi_process_adaptive(self,):
        for i,image_file in enumerate(self.image_files):
            self.process_adaptive_singlefile(image_file)

    def process_adaptive_singlefile(self,image_file):
        print(image_file)
        image = Image.open(image_file, mode='r').convert("RGB")
        gt_file_path,substr=self.get_infos(image_file,self.gts_path)
        locations = self.load_gt(gt_file_path)
        dot_map = self.generate_dot_map(locations, image.size)
        desity_map = self.adaptive_gaussian_generator(dot_map)

        np.save(os.path.join(self.density_npy_path,substr + '.npy'),desity_map)
        np.save(os.path.join(self.dot_npy_path,substr + '.npy'),dot_map)


    def multi_process_adaptive(self):
        pool = multiprocessing.Pool(processes=16)
        for file_name in self.image_files:
            pool.apply_async(self.process_adaptive_singlefile, (file_name, ))
        pool.close()
        pool.join()



if __name__=="__main__":
    shhai=ShhaiPrepare('/home/houyi/datasets/ShanghaiTech/part_B/train_data')
    # a.draw_head_dot('/home/houyi/datasets/ShanghaiTech/part_B/train_data/images/IMG_1.jpg','/home/houyi/datasets/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_1.mat')
    shhai.multi_process_adaptive()











