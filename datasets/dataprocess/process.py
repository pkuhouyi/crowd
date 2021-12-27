# -*- coding: UTF-8 -*-

from utils import *
import argparse
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--type', default='B', type=str)
parser.add_argument('--istrain', default=False, type=bool)
parser.add_argument('--basepath', default="/home/houyi/datasets/", type=str)
args = parser.parse_args()


def load_data(path):
    images = []
    densitys = []
    locations = []
    images_path = path + '/images/'
    desity_path=path+'/ada_gaussian_npy'
    dot_path=path+'/dot_npy'
    for i,image_file in enumerate(sorted(glob.glob(os.path.join(images_path, '*.jpg')))):
        print(image_file)
        image = Image.open(image_file, mode='r').convert("RGB")
        images.append(np.array(image))
        i = image_file.index('IMG_') + 4
        j = image_file.index('.jpg')
        substr = image_file[i:j]
        dot_map = np.load(os.path.join(dot_path,substr+'.npy'))
        locations.append(dot_map)
        density_map = np.load(os.path.join(desity_path,substr+'.npy'))
        densitys.append(density_map)



    print(len(images), 'loaded')
    return (images, densitys, locations)


# python process.py -basepath='/home/houyi/PycharmProjects/crowd/datasets/'
# 生成训练数据集，保存在ada_prepare_train_data目录下
# 同时生成数据集中所有原始图片的评估数据
def generate_shanghai_train(dataset_path):
    imgpath = dataset_path+'ShanghaiTech/part_'+args.type+'/train_data'

    images, ada_densitys, locations = load_data(imgpath)

    print(len(images))
    npy_path = imgpath + '/combine_val_data/'
    save_npy(npy_path, images, ada_densitys, locations)

    images, ada_densitys, locations = generate_slices(images, ada_densitys, locations)          #10倍
    # images, ada_densitys, locations=pca_augmentation(images, ada_densitys,vor_densitys, locations,times=1)   #2倍
    images, ada_densitys, locations =flip_slices(images, ada_densitys, locations)               #2倍
    images, ada_densitys, locations = shuffle_slices(images, ada_densitys, locations)
    print(len(images))

    npy_path=imgpath+'/combine_train_data/'
    save_npy(npy_path,images, ada_densitys, locations)



#没有presize和expand的原始数据集，进行了shffle
def generate_shanghai_test(dataset_path):
    imgpath = dataset_path+'ShanghaiTech/part_'+args.type+'/test_data'

    images, ada_densitys, locations = load_data(imgpath)        # 返回的是np.array类型
    print(len(images))

    npy_path=imgpath+'/combine_test_data/'
    save_npy(npy_path,images, ada_densitys, locations)


# 对shanghai数据集a_train进行处理，每张图片生成20张图片 300
if __name__ == '__main__':

    if args.istrain==True:
        generate_shanghai_train(args.basepath)
    else:
        generate_shanghai_test(args.basepath)







