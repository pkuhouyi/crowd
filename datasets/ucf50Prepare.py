
import scipy.io as sio
from datasets.basePrepare import BasePrepare
import os

class Ucf50Prepare(BasePrepare):
    def __init__(self,path, density_npy_name='/ada_gaussian_npy/',dot_npy_name='/dot_npy/'):
        images_path = path + '/images/'
        gts_path = path + '/ground-truth/'
        density_npy_path=path + density_npy_name
        dot_npy_path=path + dot_npy_name

        super(Ucf50Prepare, self).__init__(img_path=images_path,gts_path=gts_path,density_path=density_npy_path,dot_path=dot_npy_path,ext='*.jpg',)


    def load_gt(self, gtfile_path):
        dataset = sio.loadmat(gtfile_path)
        locations=dataset['annPoints']
        return locations

    def get_infos(self,image_file_name, gt_file_base_path):
        filname=os.path.basename(image_file_name)
        j = filname.index('.jpg')
        substr = filname[0:j]
        gt_file_path = gt_file_base_path + substr + '_ann.mat'
        return gt_file_path,substr




if __name__=="__main__":
    ucfcc50=Ucf50Prepare('/home/houyi/datasets/UCF_CC_50/')
    ucfcc50.multi_process_adaptive()











