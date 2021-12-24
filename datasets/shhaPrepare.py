
import scipy.io as sio
from datasets.basePrepare import BasePrepare

class ShhaiPrepare(BasePrepare):
    def __init__(self,path, density_npy_name='/ada_gaussian_npy/',dot_npy_name='/dot_npy/'):

        images_path = path + '/images/'
        gts_path = path + '/ground-truth/'
        density_npy_path=path + density_npy_name
        dot_npy_path=path + dot_npy_name

        super(ShhaiPrepare, self).__init__(img_path=images_path,gts_path=gts_path,density_path=density_npy_path,dot_path=dot_npy_path,ext='*.jpg',)

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


if __name__=="__main__":
    shhai=ShhaiPrepare('/home/houyi/datasets/ShanghaiTech/part_B/test_data')
    # a.draw_head_dot('/home/houyi/datasets/ShanghaiTech/part_B/train_data/images/IMG_1.jpg','/home/houyi/datasets/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_1.mat')
    shhai.non_multi_process_adaptive()











