import os
from torch.utils import data
from PIL import Image
from datasets.transforms import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class SHHA(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, density_transform=None, dot_transform=None):
        self.img_path = data_path + '/images'
        self.density_path = data_path + '/ada_gaussian_npy'
        self.dot_path = data_path + '/dot_npy'
        self.images_files = [filename for filename in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path,filename))]


        self.main_transform = main_transform
        self.img_transform = img_transform
        self.density_transform = density_transform
        self.dot_transform = dot_transform

    def __getitem__(self, index):
        fname = self.images_files[index]
        img, density_map, dot_map = self.read_data(fname)
        SHHA.draw_density_dot_heatmap(img, density_map, dot_map)
        if self.main_transform is not None:
            img, density_map, dot_map = self.main_transform(img, density_map, dot_map)
            SHHA.draw_density_dot_heatmap(img, density_map, dot_map)
        if self.img_transform is not None:
            img = self.img_transform(img)         
        if self.density_transform is not None:
            density_map = self.density_transform(density_map)
        if self.dot_transform is not None:
            dot_map = self.dot_transform(dot_map)
        return img, density_map, dot_map

    def __len__(self):
        return len(self.images_files)

    def read_data(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')
        i = fname.index('IMG_') + 4
        j = fname.index('.jpg')
        substr = fname[i:j]
        density_map=Image.fromarray(np.load(os.path.join(self.density_path,substr+'.npy')))
        dot_map=Image.fromarray(np.load(os.path.join(self.dot_path,substr+'.npy')))

        return img, density_map, dot_map

    def get_item(self,index=0):
        self.__getitem__(index)


    @staticmethod
    def draw_density_dot_heatmap(image,density_map,dot_map):
        density_map=np.array(density_map)
        dot_map=np.array(dot_map)
        img = image
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
        plt.imshow(dot_map,cmap=cmap,extent=None, aspect='equal')
        plt.axis('off')
        sum = np.sum(dot_map)
        print('true_count:', sum)
        plt.show()

if __name__=="__main__":

    transform=Compose([RandomHorizontallyFlip()])
    shahaia=SHHA('/home/houyi/datasets/ShanghaiTech/part_B/train_data',transform)
    shahaia.get_item()